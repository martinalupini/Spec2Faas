import os
import re
from typing import List
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_core.code_executor import CodeBlock, CodeExecutor
from .messages.MessagesTypes import *
from .utils.Utils import *


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks

class TestExecutor(RoutedAgent):
    def __init__(self, llm:str, model_client: ChatCompletionClient, code_executor: CodeExecutor) -> None:
        super().__init__("Skilled test executor")
        self._system_messages = [SystemMessage(
            content="You are a very skilled test executor."
                    "Given the function to test and the tests, create a code that will make them runnable."
                    "DO NOT MODIFY THE FUNCTION AND THE TESTS PROVIDED, just merge them in a single code."

        )]
        self._model_client = model_client
        self._code_executor = code_executor
        self._code = ""
        self._tests = ""
        self._attempts = 0
        self._max_attempts = int(os.getenv("MAX_DEBUG_ATTEMPTS"))
        self._llm = llm
        self._role = "Test Executor"
        print_green(f"Hi I'm the test executor and I use {self._llm}.")

    @message_handler
    async def handle_execute_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message from {message.sender}")
        if self._tests == "":
            self._tests = message.tests
        if self._code == "":
            self._code = message.code

        if self._tests != "" and self._code != "":
            while self._attempts < self._max_attempts:
                # Prepare input to the chat completion model.
                prompt = "Function code: " + self._code + "\nFunction tests: " + self._tests
                user_message = UserMessage(content=prompt, source="user")
                response = await self._model_client.create(
                    self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
                )

                assert isinstance(response.content, str)
                dialogue(response.content, self._role)
                code_blocks = extract_markdown_code_blocks(response.content)
                if code_blocks:
                    result = await self._code_executor.execute_code_blocks(
                        code_blocks, cancellation_token=ctx.cancellation_token
                    )
                    print_yellow(f"\n{'-' * 80}\nExecutor:\n{result.output}")

                    if "AssertionError" in result.output:
                        debug_message = await self._runtime.send_message(
                            DebugMessage(message.specification, message.code, result.output),
                            AgentId("debugger", "default"))

                        self._code = debug_message.code
                        self._attempts += 1
                    else:
                        return Message(self._code, "test_executor_response")
            # Max attempts number reached
            return Message("FAIL", "test_executor_response")
        return Message("Still waiting", "test_executor_response")
