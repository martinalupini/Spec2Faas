from unittest import TestResult

from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import UserMessage, ChatCompletionClient, SystemMessage

from experiments.MessageTypesTest import TestMessageResult
from .messages.MessagesTypes import *
from experiments.MessageTypesTest import *
from .utils.Utils import *
import time


class EntryPoint(RoutedAgent):
    def __init__(self, llm: str,  model_client: ChatCompletionClient) -> None:
        super().__init__("The entry point of the coding.")
        self._system_messages = [SystemMessage(
            content="Given the function specification create a signature for the function. "
                    "Return only the signature of the function as output."
        )]
        self._model_client = model_client
        self._llm = llm
        print_green(f"Hi I'm the entry point of the coding system and I use {self._llm}.")

    @message_handler
    async def handle_assistant_message(self, message: Message, ctx: MessageContext) -> Message:
        if message.type == "test_executor_response":
            print_green(f"{self.id.type} received message. Communicating final outcome.")
            return message

        print_green(f"{self.id.type} received message. Activating Coder and Test Designer.")

        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )
        dialogue("The function signature is: " + response.content, "Entry Point")
        await self._runtime.send_message(CodeMessage(message.content, response.content, "", "", self.id.type), AgentId("coder", "default"))
        return_message = await self._runtime.send_message(CodeMessage(message.content, response.content, "", "", self.id.type), AgentId("test_designer", "default"))

        return return_message

    @message_handler
    async def handle_test_assistant_message(self, message: TestMessage, ctx: MessageContext) -> TestMessageResult:
        print_green(f"{self.id.type} received message. Activating Coder and Test Designer.")

        start_time = time.perf_counter()
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        end_time = time.perf_counter()
        usage_metadata = response.usage
        tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens

        return TestMessageResult(response.content, end_time - start_time, tokens)

    @message_handler
    async def handle_test_system_assistant_message(self, message: TestSystemMessage, ctx: MessageContext) -> TestSystemMessage:
        if message.type == "test_executor_response":
            print_green(f"{self.id.type} received message. Communicating final outcome.")
            print_purple(str(message))
            return message

        print_green(f"{self.id.type} received message. Activating Coder and Test Designer.")
        print_purple(str(message))

        total_time = message.time
        total_tokens = message.tokens
        start_time = time.perf_counter()
        user_message = UserMessage(content=message.prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        end_time = time.perf_counter()
        usage_metadata = response.usage
        tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens
        total_time['entry_point'] = end_time - start_time
        total_tokens['entry_point'] = tokens

        return_coder = await self._runtime.send_message(TestSystemMessage(tokens= total_tokens, time = total_time,prompt = message.prompt, signature = response.content),
                                         AgentId("coder", "default"))
        return_message = await self._runtime.send_message(TestSystemMessage(tokens= return_coder.tokens, time = return_coder.time, prompt = message.prompt, signature = response.content, original_func = return_coder.original_func, code = return_coder.code),
                                         AgentId("test_designer", "default"))
        return return_message