from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId, CancellationToken
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *
from .utils.Code_Extractors import *
from experiments.MessageTypesTest import *
import ollama
import time
import os

class Coder(RoutedAgent):
    def __init__(self,llm: str, model_client: ChatCompletionClient = None, server = None) -> None:
        super().__init__("Skilled software programmer")
        self._system_messages = [SystemMessage(
            content="You are a very skilled software programmer."
                    "<TASK>"
                    "As a programmer, you are required to code a function that adheres to the function specification."
                    "Make sure to use the function signature provided."
                    "<CODE FORMATTING>"
                    "Please write code in this format:"
                    "```python"
                    "[Code]"
                    "```"
                    "</CODE FORMATTING>"
                    "<INSTRUCTIONS>"
                    "1. Make sure you understand the task."
                    "2. Decide on the most efficient way to solve the task."
                    "3. Use the function signature provided."
                    "3. Make sure your code is correct and complete."
                    "4. RETURN ONLY THE CODE OF THE FUNCTION IN THE SPECIFIED FORMAT"
                    "</INSTRUCTIONS>"
        )]
        self._model_client = model_client
        self._llm = llm
        self._text = ""
        self._role = "Software Programmer"
        self._client = None
        self._server = server
        print_green(f"Hi I'm the software programmer and I use {self._llm}.")

    @message_handler
    async def handle_generate_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Staring to generate code with {self._llm}.")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
        user_message = UserMessage(content=prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        dialogue(response.content, self._role)
        if os.getenv("UI") == "True":
            self._server.send_chunk(response.content, "coder")

        # Return message to the entry_point
        return_message = await self._runtime.send_message(
            ExecuteMessage(message.specification, response.content, "", self.id.type),
            AgentId("test_executor", "default"))
        return return_message



    @message_handler
    async def handle_generate_test_code_message(self, message: TestCodeMessage,
                                           ctx: MessageContext) -> TestCodeResult:

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
        user_message = UserMessage(content=prompt, source="user")
        if message.prompt:
            final_prompt = self._system_messages + [user_message]
        else:
            final_prompt = [user_message]
        start_time = time.perf_counter()
        response = await self._model_client.create(
            final_prompt, cancellation_token=ctx.cancellation_token
        )
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        usage_metadata = response.usage
        total_tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens

        assert isinstance(response.content, str)

        return TestCodeResult(response.content, execution_time, total_tokens, ctx.cancellation_token)

    @message_handler
    async def handle_test_system_generate_code_message(self, message: TestSystemMessage, ctx: MessageContext) -> TestSystemMessage:
        print_green(f"{self.id.type} received message. Staring to generate code with {self._llm}.")
        print_purple(str(message))

        total_time = message.time
        total_tokens = message.tokens
        start_time = time.perf_counter()
        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.prompt + "\nFunction signature: " + message.signature
        user_message = UserMessage(content=prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        end_time = time.perf_counter()
        usage_metadata = response.usage
        tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens

        # Updating time and tokens
        total_time['coder'] = end_time - start_time
        total_tokens['coder'] = tokens

        # Extracting the function string
        function_code = extract_markdown_code_blocks(response.content)
        if function_code:
            original_function = function_code[0].code
        else: original_function = response.content

        assert isinstance(response.content, str)
        # After generating the code the Coder sends a message to the TestExecutor to let it know the code
        # new_chat flag is True to make TestExecutor reset its data structures
        return_message = await self._runtime.send_message(
            TestSystemMessage(tokens = total_tokens, time = total_time, messages=message.messages +1, prompt = message.prompt, signature = message.signature, original_func = original_function, code = response.content, new_chat = True, sender = self.id.type),
            AgentId("test_executor", "default"))
        return return_message