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


def _pystring_to_tarBase64(py_code, filename= "main.py") -> str:
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

async def create_json_serverledge(code: str, name: str, runtime: str, memoryMB: int, CPUDemand: int, handler:str) -> dict:
    tar_b64 = _pystring_to_tarBase64(code, filename="main.py")

    payload = {
        "Name": name,
        "Runtime": runtime,
        "MemoryMB": memoryMB,
        "CPUDemand": CPUDemand,
        "Handler": handler,
        "TarFunctionCode": tar_b64
    }

    return {
        "payload": payload,
        "url": os.getenv("SERVERLEDGE_URL")
    }

async def register_function(url: str, payload: dict, timeout: int =20):
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            print_yellow(f"Resp status: {resp.status} Resp text: {await resp.text()}")
            try:
                # Checking if the response is an error. Is so reaise a ClientResponseError
                resp.raise_for_status()
            except aiohttp.ClientResponseError:
                text = await resp.text()
                raise RuntimeError(f"Create failed ({resp.status}): {text}")

            try:
                print("1")
                return await resp.json()
            except aiohttp.ContentTypeError:
                print("2")
                text = await resp.text()
                return {"status": "ok", "raw": text}

class FaasDeployer(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("The agents responsible for deploying the function in FaaS.")

        self._system_messages = [SystemMessage(
            content="You are an expert cloud engineer responsible for deploying Python functions to a FaaS platform."
            "Your task is to follow a strict two-step process using the available tools."
            "The input will be a Python function code."

            "Step 1: Prepare the deployment payload."
            "First, you must reformat the user's Python code into a valid handler structure. "
            "The handler function must be named 'handler' and accept 'params' and 'context' as arguments. "
            "Import necessary libraries outside the handler."
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
                "Step 2: After reformatting the code, you MUST call the `create_json_serverledge` tool to generate the deployment payload and the server URL. "
                "Choose the appropriate Name, Runtime, MemoryMB, CPUDemand and Handler."
                "The handler should be in the format function_name.handler. The examples are hello.handler and fibonacci.handler"
                "Step 3: Register the function."
                "This step is mandatory and MUST follow Step 1. Take the `payload` and `url` returned by the `create_json_serverledge` tool and use them as arguments to call the `register_function` tool."
                "The entire process is complete ONLY after the `register_function` tool has been successfully called. Do not stop after Step 1."
        )]

        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_deploy_message(self, message: DeployMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Starting to deploy the function in FaaS.")

        """
        prompt = "This is the code: " + message.code
        response = await self._model_client.create(
            self._system_messages + [UserMessage(content=prompt, source="user")], cancellation_token=ctx.cancellation_token
        )
        print_purple("The final response is: " + response.content)
        """
        # Create a session of messages.
        prompt= "This is the code: " + message.code
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=prompt, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
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

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        print_purple(create_result.content)
        # Return the result as a message.
        return Message(content=create_result.content, type="final_response")


    async def _execute_tool_call(
            self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None
        print("Executing tool " + tool.name)

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)

