from dataclasses import dataclass

from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage


@dataclass
class Message:
    content: str


class Assistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An helpful assistant ")
        self._system_messages = [SystemMessage(content="You are a translator. Your job is to translate the input phrase in English. Make sure to be as precise as possible.")]
        self._model_client = model_client

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Prepare input to the chat completion model.
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )
        # Return with the model's response.
        assert isinstance(response.content, str)
        return Message(content=response.content)
