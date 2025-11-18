from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *
from .utils.Code_Extractors import *
from experiments.MessageTypesTest import *
import time

class TestDesigner(RoutedAgent):
    def __init__(self, llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled test designer")
        self._system_messages = [SystemMessage(
            content="You are a very skilled test designer specializing in designing comprehensive and well-documented test cases for Python functions."
                    "Your goal is to generate exhaustive test cases for the given function specification, covering all relevant input scenarios and ensuring correctness, robustness, and edge coverage."
                    "<INSTRUCTIONS>"
                    "First of all, make CORRECT tests."
                    "- Focus on correctness, clarity, and diversity of test inputs."
                    "- Each test case must:"
                    "-- Follow the exact format below."
                    "-- Include a short, meaningful comment or description of what it tests."
                    "-- Be self-contained and syntactically valid Python."
                    "-- Be semantically correct with respect to the specification."
                    "- Only output test cases in the specified format Python format, nothing else."
                    "- After you have produced the tests take your time to reflect if each on of them is correct."
                    "</INSTRUCTIONS>"
                    "<OUTPUT FORMAT>"
                    "The format of test cases should be:"
                    "```python"
                    "assert function_name(input) == expected_output, 'Test Case <number>: Description'"
                    "```"
                    "</OUTPUT FORMAT>")]
        self._model_client = model_client
        self._llm = llm
        self._client = None
        self._role = "Test Designer"
        print_green(f"Hi I'm the test designer and I use {self._llm}.")

    @message_handler
    async def handle_generate_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Creating now a test suite for the function.")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature + "\nPlease generate the tests:"
        user_message = UserMessage(content=prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        dialogue(response.content, self._role)

        # Sending a message to TestExecutor
        return_message = await self._runtime.send_message(
            CodeMessage(message.specification, message.function_signature, "", response.content, self.id.type),
            AgentId("test_executor", "default"))
        return return_message

    @message_handler
    async def handle_generate_test_code_message(self, message: TestCodeMessage, ctx: MessageContext) -> TestCodeResult:
        print_green(f"{self.id.type} received message. Creating now a test suite for the function.")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature + "\nPlease generate the tests:"
        user_message = UserMessage(content=prompt, source="user")
        start_time = time.perf_counter()
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        usage_metadata = response.usage
        tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens
        end_time = time.perf_counter()
        return TestCodeResult(response.content, end_time - start_time, tokens, ctx.cancellation_token)

    @message_handler
    async def handle_test_system_generate_code_message(self, message: TestSystemMessage, ctx: MessageContext) -> TestSystemMessage:
        print_green(f"{self.id.type} received message. Creating now a test suite for the function.")
        print_purple(str(message))

        # Retrieving time and tokens from the message
        total_tokens = message.tokens
        total_time = message.time

        start_time = time.perf_counter()
        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.prompt + "\nFunction signature: " + message.signature + "\nPlease generate the tests:"
        user_message = UserMessage(content=prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        end_time = time.perf_counter()
        usage_metadata = response.usage
        tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens

        # Updating time and tokens
        total_time['test_designer'] = end_time - start_time
        total_tokens['test_designer'] = tokens

        # Not only passing the original response to test executor but also the tests as a string
        test_code = extract_markdown_code_blocks(response.content)
        if test_code:
            tests = test_code[0].code
        else:
            tests = response.content

        # Sending a message to TestExecutor
        return_message = await self._runtime.send_message(
            TestSystemMessage(tokens = total_tokens, time = total_time, messages=message.messages +1, prompt = message.prompt, signature = message.signature, original_func = message.original_func, code = message.code, tests = response.content, tests_str = tests , sender = self.id.type),
            AgentId("test_executor", "default"))
        return return_message
