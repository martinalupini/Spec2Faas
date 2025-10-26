from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId, default_subscription, TopicId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *


@default_subscription()
class TestDesigner(RoutedAgent):
    def __init__(self, llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled test designer")
        self._system_messages = [SystemMessage(
            content= "You are a very skilled software tester specializing in designing comprehensive and well-documented test cases for Python functions."
            "Your goal is to generate exhaustive test cases for the given function specification, covering all relevant input scenarios and ensuring correctness, robustness, and edge coverage."
                     "GUIDELINES"
                     "-Focus on correctness, clarity, and diversity of test inputs."
                     "- Each test case must:"
                     "- Follow the exact format below."
                     "- Include a short, meaningful comment or description of what it tests."
                     "- Be self-contained and syntactically valid Python."
                     "OUTPUT FORMAT:"
                     "- The format of test cases should be:"
                     "```python"
                     "assert function_name(input) == expected_output, 'Test Case Description'"
                     "```"
                     "IMPORTANT RULE: "
                     "-Only output test cases in the specified format Python format, nothing else."
                     "-If the specification is not complete make REASONABLE assumptions on how the function should behave."
        )]
        self._model_client = model_client
        self._llm = llm
        self._role = "Test Designer"
        print_green(f"Hi I'm the test designer and I use {self._llm}.")

    @message_handler
    async def handle_generate_code_message(self, message: CodeMessage, ctx: MessageContext) -> None:
        print_green(f"{self.id.type} received message. Creating now a test suite for the function.")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
        user_message = UserMessage(content=prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        dialogue(response.content, self._role)
        await self.publish_message(
            ExecuteCodeRequest(message.specification, "", response.content, self.id.type),
            topic_id=TopicId("default", self.id.key))
