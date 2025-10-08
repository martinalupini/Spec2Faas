from dataclasses import dataclass
from autogen_core import MessageContext, RoutedAgent, message_handler


@dataclass
class MyMessageType:
    content: str


class Assistant(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Assistant")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")