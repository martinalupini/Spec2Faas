from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *
import ollama

class TestDesigner(RoutedAgent):
    def __init__(self, llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled test designer")
        self._system_prompt = """You are a very skilled test designer specializing in designing comprehensive and well-documented test cases for Python functions.
                    Your goal is to generate exhaustive test cases for the given function specification, covering all relevant input scenarios and ensuring correctness, robustness, and edge coverage.
                    <INSTRUCTIONS>
                    First of all, make CORRECT tests.
                    - If the specification is not complete make REASONABLE assumptions on how the function should behave.
                    - Focus on correctness, clarity, and diversity of test inputs.
                    - Each test case must:
                    -- Follow the exact format below.
                    -- Include a short, meaningful comment or description of what it tests.
                    -- Be self-contained and syntactically valid Python.
                    -- Be semantically correct with respect to the specification.
                    - Only output test cases in the specified format Python format, nothing else.
                    - After you have produced the tests take your time to reflect if each on of them is correct.
                    </INSTRUCTIONS>
                    <OUTPUT FORMAT>
                    The format of test cases should be:
                    ```python
                    assert function_name(input) == expected_output, 'Test Case <number>: Description'
                    ```
                    </OUTPUT FORMAT>\n"""
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

        if self._llm == "gemini-2.5-pro" or self._llm == "gemini-2.0-flash":
            # Prepare input to the chat completion model.
            prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature + "\nPlease generate the tests:"
            user_message = UserMessage(content=prompt, source="user")
            response = await self._model_client.create(
                self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
            )

            assert isinstance(response.content, str)
            dialogue(response.content, self._role)
            return_message = await self._runtime.send_message(
                CodeMessage(message.specification, message.function_signature, "", response.content, self.id.type),
                AgentId("test_executor", "default"))
            return return_message
        else:

            if self._client is None:
                self._client = ollama.Client(host='http://160.80.97.151:11434')

            # Prepare input to the chat completion model.
            prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
            response = self._client.chat(
                model=self._llm,
                messages=[
                    {'role': 'user',
                     'content':self._system_prompt + prompt},
                ]
            )

            dialogue(response['message']['content'], self._role)
            return_message = await self._runtime.send_message(
                CodeMessage(message.specification, message.function_signature, "", response['message']['content'],
                            self.id.type), AgentId("test_executor", "default"))
            return return_message

