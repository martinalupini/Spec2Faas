from autogen_core import MessageContext, RoutedAgent, message_handler,AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .coding_agents.messages.MessagesTypes import *
from .coding_agents.utils.Utils import *

class Assistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An helpful assistant ")
        self._system_messages = [SystemMessage(
            content="You are the entry point to a code generator and deployment app."
                    "Inspect the input received by the user.:"
                    "1.If it contains a specification of a function or it asks to code a function, translate it in English and return the translation. "
                    "-Make the translation clear and direct, so that an AI assistant can easily generate code from it."
                    "-Put the word 'translation' as incipit of the text. Return only one translation."
                    "-If it is already in English don't translate."
                    "-Be as precise as possible with the translation."
                    "2. If it contains a function already complete return the word 'deployment' followed by the function specified by the user."
                    "3. If the input does not fall in the previous two categories or the specification is unsure asks for a clarification."
                    "-If the user continue to provide an input that is neither a function specification nor a function code continue to ask for clarification."
        )]
        self._model_client = model_client

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
            #print("Assistant deployment")
            await self._runtime.send_message(DeployMessage(code=response.content.removeprefix("deployment:")),AgentId("faas_deployer", "default"))
            return Message(content="The function is successfully deployed.", type="deployment")
        elif response.content.startswith("translation"):
            #print("Assistant translation")
            # The translation is complete so we can send a message to the Coder and the TestDesigner
            return_message = await self._runtime.send_message(Message(response.content.removeprefix("translation:"), type="request"), AgentId("entry_point", "default"))
            if return_message.content == "FAIL":
                return Message(content="We couldn't generate a correct function given the specification.", type="failure")
            else:
                await self._runtime.send_message(DeployMessage(code=return_message.content),
                                                 AgentId("faas_deployer", "default"))
                return Message(content="The function is successfully deployed.", type="deployment")
        else:
            # We need more context from the user
            return Message(content=response.content, type="request")
