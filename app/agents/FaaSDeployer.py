import os
import io
import tarfile
import base64
import time
import json
import aiohttp
import asyncio
from typing import List

from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId, CancellationToken, FunctionCall
from autogen_core.models import LLMMessage, AssistantMessage, FunctionExecutionResultMessage, FunctionExecutionResult

from .coding_agents.TestDesigner import *
from .coding_agents.messages.MessagesTypes import *
from .coding_agents.utils.Utils import *
from autogen_core.tools import FunctionTool, Tool


def _pystring_to_tarBase64(py_code, filename) -> str:
    data = py_code.encode("utf-8")
    mem = io.BytesIO()

    with tarfile.open(fileobj=mem, mode="w") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        info.mtime = int(time.time())
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        tar.addfile(info, io.BytesIO(data))

    mem.seek(0)
    return base64.b64encode(mem.getvalue()).decode("ascii")

async def create_json_serverledge(code: str, name: str, runtime: str, memoryMB: int, CPUDemand: int) -> dict:
    print_yellow("\nThis is the name:\n" + name)
    print_yellow("This is the code:\n" + code)

    filename = name +".py"
    tar_b64 = _pystring_to_tarBase64(code, filename=filename)
    handler = name + ".handler"

    payload = {
        "Name": name,
        "Runtime": runtime,
        "MemoryMB": memoryMB,
        "CPUDemand": CPUDemand,
        "Handler": handler,
        "TarFunctionCode": tar_b64
    }

    headers = {"Content-Type": "application/json"}
    url = os.getenv("SERVERLEDGE_URL")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            print_yellow(f"Resp status: {resp.status} Resp text: {await resp.text()}")
            if resp.ok:
                # If there are no errors content-type is json otherwise is text
                return await resp.json()
            else:
                text = await resp.text()
                return {"status": resp.status, "text": text}


class FaasDeployer(RoutedAgent):
    def __init__(self, llm: str, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("The agents responsible for deploying the function in FaaS.")

        self._system_messages = [SystemMessage(
            content="You are an expert cloud engineer responsible for deploying Python functions to a FaaS platform."
            "Your task is to follow a strict two-step process using the available tools."
            "The input will be a Python function code."
            "<INSTRUCTIONS>"
            "Step 1: Prepare the deployment payload."
            "First, you must reformat the user's Python code into a valid handler structure. Do not modify the user's code provided but just add the handler."
            "Import necessary libraries outside the handler."
            "Look at the examples in <EXAMPLES>"
            "Step 2: After reformatting the code, you MUST call the `create_json_serverledge` tool to register the function. "
            "Choose the appropriate Name, Runtime (default is python310), MemoryMB, CPUDemand and Handler."
            "The handler should invoke the function and return the result. The definition of the function has to be outside of the handler. Look carefully at the examples provided."
            "The handler should return a dictionary."
            "The handler receives input as a string. Make sure to cast the input to the appropriate data type (e.g., int, float, bool, etc.) based on the expected arguments of the target function before calling it."
            "If you obtain errors from the server reflect on these errors. For example if the error is 404 'Chosen runtime does not exists' choose python310. "
            "If the error is 409 'Function already exists' choose another name for the function."
            "Only stop invoking tools when the server response is 200."
            "</INSTRUCTIONS>"
            "<EXAMPLES>"
            "Example 1: "
            "def handler(params, context):"
                "n = params['n']"
                "return ''.join(fibonacci_nums(int(n)))"
            "def fibonacci_nums(n):"
                "sequence = \"\""
                "if n <= 0:"
                    "sequence += \"0\""
                    "return sequence"
                "sequence = \"0, 1\""
                "count = 2"
                "n1 = 0"
                "n2 = 1"
                "while count <= n:"
                    "next_value = n2 + n1"
                    "sequence += \",\" + \"\".join(str(next_value))"
                    "n1 = n2"
                    "n2 = next_value"
                    "count += 1"
                "return sequence"
                "Example 2:"
                "def handler(params, context):"
                    "print(\"Executing function....\")"
                    "return \"Hello, Serverledge!\nParams: {}\".format(params)"
                "Example 3:"
                "import re"
                "def handler(params, context):"
                    "return grep(\"grep\", params[\"InputText\"])"
                "def grep(pattern, text):"
                    "lines = text.split('\n')"
                    "result = [line for line in lines if re.search(pattern, line)]"
                    "return '\n'.join(result)"
                "</EXAMPLES>"
        )]

        self._model_client = model_client
        self._tools = tool_schema
        self._llm = llm
        self._role = "FaaS Deployer"
        print_green(f"Hi I'm the debugger and I use {self._llm}.")

    @message_handler
    async def handle_deploy_message(self, message: DeployMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Starting to deploy the function in FaaS.")

        # The following code follows the guide at https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/components/tools.html#tool-equipped-agent
        # Create a session of messages.
        prompt= "This is the code: " + message.code
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=prompt, source="user")]

        while True:
            # Run the chat completion with the tools.
            create_result = await self._model_client.create(
                messages=session,
                tools=self._tools,
                cancellation_token=ctx.cancellation_token,
            )

            # If there are no tool calls, return the result.
            if isinstance(create_result.content, str):
                dialogue(create_result.content, self._role)
                return Message(content=create_result.content, type = "final_response")
            assert isinstance(create_result.content, list) and all(
                isinstance(call, FunctionCall) for call in create_result.content
            )

            # Add the first model create result to the session.
            session.append(AssistantMessage(content=create_result.content, source="assistant"))

            # Execute the tool calls.
            results = await asyncio.gather(
                *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
            )

            # Add the function execution results to the session.
            session.append(FunctionExecutionResultMessage(content=results))


    async def _execute_tool_call(
            self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None
        dialogue("Executing tool " + tool.name, self._role)

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)

