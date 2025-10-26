from autogen_core import MessageContext, RoutedAgent, message_handler,AgentId, default_subscription, TopicId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .coding_agents.messages.MessagesTypes import *
from .coding_agents.utils.Utils import *


@default_subscription()
class Assistant(RoutedAgent):
    def __init__(self, llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("An helpful assistant ")
        self._system_messages = [SystemMessage(
            content="You are the entry point to a code generator and deployment app."
                    "Inspect the input received by the user.:"
                    "1.If it contains a specification of a function or it asks to code a function, translate it in English and return the translation. "
                    "-Make the translation clear and direct, so that an AI assistant can easily generate code from it."
                    "-Put the word 'translation' as incipit of the text. Return only one translation."
                    "-If it is already in English don't translate, but still put translation as incipit of text."
                    "-Be as precise as possible with the translation."
                    "2. If it contains a function already complete return the word 'deployment' followed by the function specified by the user."
                    "3. If the input does not fall in the previous two categories or the specification is unsure asks for a clarification."
                    "-If the user continue to provide an input that is neither a function specification nor a function code continue to ask for clarification."
        )]
        self._model_client = model_client
        self._llm = llm
        self._text = ""
        self._role = "Assistant"
        print_green(f"Hi I'm the assistant and I use {self._llm}.")

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Staring to analyze user's prompt.")
        # Prepare input to the chat completion model.
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )


        assert isinstance(response.content, str)
        if response.content.startswith("deployment"):
            dialogue(response.content, self._role)
            await self.publish_message(DeployMessage(code=response.content.removeprefix("deployment:")), topic_id=TopicId("default", self.id.key))
            return Message(content="The function is being deployed.", type="deployment")
        elif response.content.startswith("translation"):
            dialogue(response.content, self._role)
            # The translation is complete so we can send a message to the Coder and the TestDesigner
            await self.publish_message(CodeWritingRequest(response.content.removeprefix("translation:")), topic_id=TopicId("default", self.id.key))
            return Message(content="The function is being created and deployed.", type="deployment")

        else:
            # We need more context from the user
            dialogue(response.content, self._role)
            return Message(content=response.content, type="request")

    @message_handler
    async def handle_code_message(self, message: FinalCodeWritingResult, ctx: MessageContext) -> None:
        print_green(f"{self.id.type} received message. Deploying the code.")

        if message.type == "deployment":
            await self.publish_message(DeployMessage(code=message.code), topic_id=TopicId("default", self.id.key))
        else:
            dialogue("It was not possible to generate a correct code.", self._role)

