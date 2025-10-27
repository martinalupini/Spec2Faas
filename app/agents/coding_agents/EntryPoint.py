from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId, default_subscription, TopicId
from autogen_core.models import UserMessage, ChatCompletionClient, SystemMessage
from .messages.MessagesTypes import *
from .utils.Utils import *


@default_subscription()
class EntryPoint(RoutedAgent):
    def __init__(self, llm: str,  model_client: ChatCompletionClient) -> None:
        super().__init__("The entry point of the coding.")
        self._system_messages = [SystemMessage(
            content="Given the function specification create a signature for the function. "
                    "Return only the signature of the function as output."
        )]
        self._model_client = model_client
        self._llm = llm

    @message_handler
    async def handle_assistant_message(self, message: CodeWritingRequest, ctx: MessageContext) -> None:
        print_green(f"{self.id.type} received message. Activating Coder and Test Designer.")

        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )
        dialogue("The function signature is: " + response.content, "Entry Point")
        await self.publish_message(CodeMessage(message.content, response.content), topic_id=TopicId("default", self.id.key))


    @message_handler
    async def handle_executor_message(self, message: CodeExecutorFinalResult, ctx: MessageContext) -> None:
        print_green(f"{self.id.type} received message. Communicating final outcome.")
        if message.type == "fail":
            await self.publish_message(FinalCodeWritingResult(code=message.code, type="fail"), topic_id=TopicId("default", self.id.key))
        else:
            await self.publish_message(FinalCodeWritingResult(code=message.code, type="deployment"), topic_id=TopicId("default", self.id.key))

    @message_handler
    async def handle_hello_message(self, message: HelloMessage, ctx: MessageContext) -> None:
        print_purple(f"Hi I'm the {self._role} and I use {self._llm}.")