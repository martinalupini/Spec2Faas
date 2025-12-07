from typing import List
import re
import os
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, LLMMessage, AssistantMessage
from .messages.MessagesTypes import *
from experiments.MessageTypesTest import *
from .utils.Utils import *

class Debugger(RoutedAgent):
    def __init__(self,llm: str, model_client: ChatCompletionClient, server = None) -> None:
        super().__init__("Skilled software debugger")
        self._system_messages = [SystemMessage(content="You are a very skilled software programmer specialized in debugging code."
                    "As a debugger, you are required to correct the code given as input"
                    "<INSTRUCTIONS>"
                    "Step 1: inspect the code in input and the error message."
                    "Step 2: explain the error message and understand why the code is not working"
                    "Step 3: correct the code."
                    "Return ONLY the code of the function in the specified format."
                    "It is important to remember previous corrections of the same function."
                    "</INSTRUCTIONS>"
                    "<OUTPUT FORMAT>"
                    "```python"
                    "[Code]"
                    "```"
                    "</OUTPUT FORMAT>"
        )]
        self._model_client = model_client
        # Adding a memory so that the debugger as an history of what happend
        self._debug_chat: List[LLMMessage] = self._system_messages
        self._counter = 0
        self._llm = llm
        self._role = "Debugger"
        self._server = server
        print_green(f"Hi I'm the debugger and I use {self._llm}.")

    @message_handler
    async def handle_debug_code_message(self, message: DebugMessage, ctx: MessageContext) -> DebugMessage:
        self._counter += 1
        print_green(f"{self.id.type} received message. Staring to debug code. Attempt {self._counter}")

        # Prepare input to the chat completion model.
        prompt = "This is the function specification: " + message.specification + "\nThis is the function code: " + message.code + "\nAnd this is the error message: " + message.error_message + "\nCan you correct the code?"
        self._debug_chat.append(UserMessage(content=prompt, source="user"))
        response = await self._model_client.create(
            messages=self._debug_chat, cancellation_token=ctx.cancellation_token
        )
        self._debug_chat.append(AssistantMessage(content=response.content, source="assistant"))

        assert isinstance(response.content, str)
        dialogue(response.content, self._role)
        if os.getenv("UI") == "True":
            self._server.send_chunk(response.content, "debugger")

        # Extract the markdown code
        match = re.search(r"(```python\n.*?```)", response.content, re.DOTALL)
        code_block = match.group(1).strip()

        # Respond back to the TestExecutor
        return DebugMessage(message.specification, code_block, "")

    @message_handler
    async def handle_test_debug_code_message(self, message: TestDebugMessage, ctx: MessageContext) -> TestDebugResult:
        # In the tests the Debugger is called multiple times.
        # The data structures need to be reset at each new entry of the dataset
        if message.new_chat:
            self._counter = 0
            self._debug_chat: List[LLMMessage] = self._system_messages
        else:
            self._counter += 1
        print_green(f"{self.id.type} received message. Attempt {self._counter}")
        print_purple(str(message))

        # Prepare input to the chat completion model.
        prompt = "This is the function specification: " + message.specification + "\nThis is the function code: " + message.code + "\nAnd this is the error message: " + message.error_message + "\nCan you correct the code?"

        try:
            self._debug_chat.append(UserMessage(content=prompt, source="user"))

            response = await self._model_client.create(
                messages=self._debug_chat, cancellation_token=ctx.cancellation_token
            )
            self._debug_chat.append(AssistantMessage(content=response.content, source="assistant"))

            usage_metadata = response.usage
            tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens

            assert isinstance(response.content, str)

            # Extract the markdown code
            match = re.search(r"(```python\n.*?```)", response.content, re.DOTALL)
            code_block = match.group(1).strip()

            # Respond back to the TestExecutor
            return TestDebugResult(code_block, tokens)
        except Exception as e:

            # If an exception occurs (i.e. maximum number of tokens) returns an empty string
            return TestDebugResult("", 1048575)

