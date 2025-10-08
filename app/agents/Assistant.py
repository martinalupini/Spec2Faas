from dataclasses import dataclass

from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage


@dataclass
class Message:
    content: str
    type: str


class Assistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An helpful assistant ")
        self._system_messages = [SystemMessage(
            content="You are the entry point to a code generator and deployment app."
                    "This is what you have to do:"
                    "1. Inspect the input received by the user. If it contains a specification of a function to code, translate it in English and return the translation with the word 'translation:' as incipit. "
                    "Be as precise as possible with the translation."
                    "2. If it contains a function to deliver return the word 'deployment'."
                    "3. If the input does not fall in the previous two categories or the specification is unsure asks for a clarification."
                    "If the user continue to provide an input that is neither a function specification nor a function code continue to ask for clarification."
        )]
        self._model_client = model_client

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Prepare input to the chat completion model.
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        if response.content.startswith("deployment"):
            return Message(content=response.content, type="deployment")
        elif response.content.startswith("translation"):
            return Message(content=response.content, type="translation")
        else:
            return Message(content=response.content, type="request")
