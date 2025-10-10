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
    def __init__(self, model_client: ChatCompletionClient, code_executor: CodeExecutor) -> None:
        super().__init__("Skilled test executor")
        self._system_messages = [SystemMessage(
            content="You are a very skilled test executor."
                    "Given the function to test and the tests, create a code that will make them runnable."

        )]
        self._model_client = model_client
        self._code_executor = code_executor
        self._code = ""
        self._tests = ""

    @message_handler
    async def handle_execute_code_message(self, message: CodeMessage, ctx: MessageContext) -> None:
        print_green(f"{self.id.type} received message from {message.sender}")
        if self._tests == "":
            self._tests = message.tests
        if self._code == "":
            self._code = message.code

        if self._tests != "" and self._code != "":
            # Prepare input to the chat completion model.
            prompt = "Function code: " + self._code + "\nFunction tests: " + self._tests
            user_message = UserMessage(content=prompt, source="user")
            response = await self._model_client.create(
                self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
            )

            assert isinstance(response.content, str)
            print_purple(response.content)
            code_blocks = extract_markdown_code_blocks(response.content)
            if code_blocks:
                result = await self._code_executor.execute_code_blocks(
                    code_blocks, cancellation_token=ctx.cancellation_token
                )
                print_yellow(f"\n{'-' * 80}\nExecutor:\n{result.output}")


