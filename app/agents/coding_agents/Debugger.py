from typing import List
import re
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, LLMMessage, AssistantMessage
from .messages.MessagesTypes import *
from .utils.Utils import *

class Debugger(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled software debugger")
        self._system_messages = [SystemMessage(content="You are a very skilled software programmer specialized in debugging code."
                    "**Task**: As a debugger, you are required to correct the code given as input"
                    "Step 1: inspect the code in input and the error message."
                    "Step 2: explain the error message and understand why the code is not working"
                    "Step 3: correct the code."
                    "Return ONLY the code of the function in the format format:"
                    "```python"
                    "[Code]"
                    "```"
        )]
        self._model_client = model_client
        # Adding a memory so that the debugger as an history of what happend
        self._debug_chat: List[LLMMessage] = self._system_messages
        self._counter = 0

    @message_handler
    async def handle_debug_code_message(self, message: DebugMessage, ctx: MessageContext) -> DebugMessage:
        self._counter += 1
        print_green(f"{self.id.type} received message. Staring to debug code. Attempt {self._counter}")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction code: " + message.code + " Error message: " + message.error_message
        self._debug_chat.append(UserMessage(content=prompt, source="user"))
        response = await self._model_client.create(
            messages=self._debug_chat, cancellation_token=ctx.cancellation_token
        )
        self._debug_chat.append(AssistantMessage(content=response.content, source="assistant"))

        assert isinstance(response.content, str)
        print_purple(response.content)

        match = re.search(r"(```python\n.*?```)", response.content, re.DOTALL)
        code_block = match.group(1).strip()

        return DebugMessage(message.specification, code_block, "")

