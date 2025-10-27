from typing import List
import re
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId, default_subscription, TopicId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, LLMMessage, AssistantMessage
from .messages.MessagesTypes import *
from .utils.Utils import *

@default_subscription()
class Debugger(RoutedAgent):
    def __init__(self,llm: str, model_client: ChatCompletionClient) -> None:
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
                    "It is important to remember previous corrections of the same function."
        )]
        self._model_client = model_client
        # Adding a memory so that the debugger as an history of what happend
        self._debug_chat: List[LLMMessage] = self._system_messages
        self._counter = 0
        self._llm = llm
        self._role = "Debugger"

    @message_handler
    async def handle_debug_code_message(self, message: DebugMessage, ctx: MessageContext) -> None:
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

        match = re.search(r"(```python\n.*?```)", response.content, re.DOTALL)
        code_block = match.group(1).strip()

        await self.publish_message(
            ExecuteCodeRequest(message.specification, code_block, message.tests,  self.id.type),
            topic_id=TopicId("default", self.id.key))

    @message_handler
    async def handle_hello_message(self, message: HelloMessage, ctx: MessageContext) -> None:
        print_purple(f"Hi I'm the {self._role} and I use {self._llm}.")