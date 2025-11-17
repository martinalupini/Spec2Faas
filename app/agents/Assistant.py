from autogen_core import MessageContext, RoutedAgent, message_handler,AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .coding_agents.messages.MessagesTypes import *
from experiments.MessageTypesTest import *
from .coding_agents.utils.Utils import *
import time

class Assistant(RoutedAgent):
    def __init__(self, llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("An helpful assistant ")
        self._system_messages = [SystemMessage(
            content="You are the entry point to a code generator and deployment app."
                    "<INSTRUCTIONS>"
                    "Inspect the input received by the user:"
                    "1.If it contains a specification of a function or it asks to code a function, translate it in English and return the translation. "
                    "-Make the translation clear and direct, so that an AI assistant can easily generate code from it."
                    "-Put the word 'translation' as incipit of the text. Return only one translation."
                    "-If it is already in English don't translate, but still put translation as incipit of text."
                    "-Be as precise as possible with the translation."
                    "2. If it contains a function already complete return the word 'deployment' followed by the function specified by the user."
                    "3. If the input does not fall in the previous two categories or the specification is unsure asks for a clarification."
                    "-If the user continue to provide an input that is neither a function specification nor a function code continue to ask for clarification."
                    "Be careful when inspecting user prompt."
                    "</INSTRUCTIONS>"
        )]

        self._system_messages_test = [SystemMessage(
            content="You are the entry point to a code generator and deployment app."
                    "<INSTRUCTIONS>"
                    "Inspect the input received by the user:"
                    "1.If it contains a specification of a function or it asks to code a function, translate it in English and return the translation. "
                    "-Make the translation clear and direct, so that an AI assistant can easily generate code from it."
                    "-Return only one translation."
                    "-If it is already in English don't translate."
                    "-Be as precise as possible with the translation."
                    "Be careful when inspecting user prompt."
                    "</INSTRUCTIONS>"
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
            await self._runtime.send_message(DeployMessage(code=response.content.removeprefix("deployment:")),AgentId("faas_deployer", "default"))
            return Message(content="The function is successfully deployed.", type="deployment")
        elif response.content.startswith("translation"):
            dialogue(response.content, self._role)
            # The translation is complete so we can send a message to the Coder and the TestDesigner
            return_message = await self._runtime.send_message(Message(response.content.removeprefix("translation:"), type="request"), AgentId("entry_point", "default"))
            if return_message.content == "FAIL":
                return Message(content="We couldn't generate a correct function given the specification.", type="failure")
            else:
                mess = await self._runtime.send_message(DeployMessage(code=return_message.content),
                                                 AgentId("faas_deployer", "default"))
                return Message(content=mess.content, type="deployment")
        else:
            # We need more context from the user
            dialogue(response.content, self._role)
            return Message(content=response.content, type="request")

    @message_handler
    async def handle_test_user_message(self, message: TestMessage, ctx: MessageContext) -> TestMessageResult:
        print_green(f"{self.id.type} received message. Staring to analyze user's prompt.")
        # Prepare input to the chat completion model.
        start_time = time.perf_counter()
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        usage_metadata = response.usage
        tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens

        assert isinstance(response.content, str)

        end_time = time.perf_counter()
        return TestMessageResult(response.content.removeprefix("translation:"), end_time - start_time, tokens)


