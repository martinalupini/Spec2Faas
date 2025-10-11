from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import UserMessage, ChatCompletionClient, SystemMessage
from .messages.MessagesTypes import *
from .utils.Utils import *


class EntryPoint(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("The entry point of the coding.")
        self._system_messages = [SystemMessage(
            content="Given the function specification create a signature for the function. "
                    "Return only the signature of the function as output."
        )]
        self._model_client = model_client

    @message_handler
    async def handle_assistant_message(self, message: Message, ctx: MessageContext) -> Message:
        if message.type == "test_executor_response":
            print_green(f"{self.id.type} received message. Communicating final outcome.")
            return message

        print_green(f"{self.id.type} received message. Activating Coder and Test Designer.")

        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )
        print_purple("The function signature is: " + response.content)
        await self._runtime.send_message(CodeMessage(message.content, response.content, "", "", self.id.type), AgentId("coder", "default"))
        return_message = await self._runtime.send_message(CodeMessage(message.content, response.content, "", "", self.id.type), AgentId("test_designer", "default"))

        return return_message