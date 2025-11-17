import os
import re
import time
from typing import List
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_core.code_executor import CodeBlock, CodeExecutor
from .messages.MessagesTypes import *
from experiments.MessageTypesTest import *
from .utils.Utils import *
from .utils.Code_Extractors import *

class TestExecutor(RoutedAgent):
    def __init__(self, llm:str, model_client: ChatCompletionClient, code_executor: CodeExecutor) -> None:
        super().__init__("Skilled test executor")
        self._system_messages = [SystemMessage(
            content="You are a very skilled test executor."
                    "Given the function to test and the tests, create a code that will make them runnable."
                    "<INSTRUCTIONS>"
                    "DO NOT MODIFY THE FUNCTION AND THE TESTS PROVIDED, just merge them in a single code."
                    "</INSTRUCTIONS>"

        )]
        self._model_client = model_client
        self._code_executor = code_executor
        self._code = ""
        self._tests = ""
        self._attempts = 0
        self._max_attempts = int(os.getenv("MAX_DEBUG_ATTEMPTS"))
        self._llm = llm
        self._role = "Test Executor"
        print_green(f"Hi I'm the test executor and I use {self._llm}.")

    @message_handler
    async def handle_execute_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message from {message.sender}")
        if self._tests == "":
            self._tests = message.tests
        if self._code == "":
            self._code = message.code

        if self._tests != "" and self._code != "":
            while self._attempts < self._max_attempts:
                # Prepare input to the chat completion model.
                prompt = "Function code: " + self._code + "\nFunction tests: " + self._tests
                user_message = UserMessage(content=prompt, source="user")
                response = await self._model_client.create(
                    self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
                )

                assert isinstance(response.content, str)
                dialogue(response.content, self._role)
                code_blocks = extract_markdown_code_blocks(response.content)
                if code_blocks:
                    result = await self._code_executor.execute_code_blocks(
                        code_blocks, cancellation_token=ctx.cancellation_token
                    )
                    if result.output == "":
                        print_yellow(f"\n{'-' * 130}\nExecutor:\nThe function passes all the tests.\n{'-' * 130}")
                    else:
                        print_yellow(f"\n{'-' * 130}\nExecutor:\n{result.output}\n{'-' * 130}")

                    if "AssertionError" in result.output:
                        debug_message = await self._runtime.send_message(
                            DebugMessage(message.specification, message.code, result.output),
                            AgentId("debugger", "default"))

                        self._code = debug_message.code
                        self._attempts += 1
                    else:
                        return Message(self._code, "test_executor_response")
            # Max attempts number reached
            return Message("FAIL", "test_executor_response")
        return Message("Still waiting", "test_executor_response")

    @message_handler
    async def handle_test_execute_code_message(self, message: TestExecCodeMessage, ctx: MessageContext) -> TestExecCodeResult:
        print_green(f"{self.id.type} received message.")
        self._tests = message.tests
        self._code = message.code
        entry_point = message.function_signature
        tokens_executor = 0
        time_executor = 0
        tokens_debugger = 0
        time_debugger = 0
        first_chat = True
        self._attempts = 0


        if message.system:
            while self._attempts < self._max_attempts:
                start_time = time.perf_counter()
                # Prepare input to the chat completion model.
                prompt = "Function code: " + self._code + "\nFunction tests: " + self._tests
                user_message = UserMessage(content=prompt, source="user")
                response = await self._model_client.create(
                    self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
                )

                usage_metadata = response.usage
                tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens
                tokens_executor += tokens

                assert isinstance(response.content, str)
                print(response.content)
                code_blocks = extract_markdown_code_blocks(response.content)
                if code_blocks:
                    result = await self._code_executor.execute_code_blocks(
                        code_blocks, cancellation_token=ctx.cancellation_token
                    )
                    end_time = time.perf_counter()
                    time_executor += end_time - start_time
                    if "AssertionError" in result.output or "Error" in result.output:
                        start_time_debug = time.perf_counter()
                        debug_message = await self._runtime.send_message(
                            TestDebugMessage(message.specification, message.code, result.output, first_chat),
                            AgentId("debugger", "default"))
                        end_time_debug = time.perf_counter()
                        tokens_debugger += end_time_debug - start_time_debug
                        time_debugger += end_time_debug - start_time_debug

                        if debug_message.code == "":
                            return TestExecCodeResult(final_function=message.code, passed=False, time=time_executor,
                                                      tokens=tokens_executor, attempts=self._max_attempts, tokens_debugger=tokens_debugger, time_debugger=time_debugger)

                        self._code = debug_message.code
                        self._attempts += 1
                        first_chat = False
                    else:
                        return TestExecCodeResult(final_function=self._code, passed=True, time=time_executor,
                                                  tokens=tokens_executor, attempts=self._attempts, tokens_debugger=tokens_debugger,time_debugger=time_debugger)
            # Max attempts number reached
            return TestExecCodeResult(final_function=message.code, passed=False, time= time_executor, tokens = tokens_executor, attempts=self._max_attempts, tokens_debugger=tokens_debugger,time_debugger=time_debugger)
        else:
            start_time = time.perf_counter()
            while self._attempts < self._max_attempts:

                dependencies = "```sh\npip install numpy\n```"
                code_blocks = extract_markdown_code_blocks(dependencies)

                code = self._code + self._tests + "\n\ncheck(" + entry_point + ")\n"
                invocation_code = CodeBlock(code=code, language='python')
                code_blocks.append(invocation_code)

                result = await self._code_executor.execute_code_blocks(
                    code_blocks, cancellation_token=ctx.cancellation_token
                )

                print_yellow(f"\n{'-' * 130}\nExecutor:\n{result.output}\n{'-' * 130}")

                if "AssertionError" in result.output or "Error" in result.output:
                    debug_message = await self._runtime.send_message(
                        TestDebugMessage(message.specification, message.code, result.output, first_chat),
                        AgentId("debugger", "default"))

                    tokens += debug_message.tokens

                    if debug_message.code == "":
                        end_time = time.perf_counter()
                        return TestExecCodeResult(final_function="", passed=False, time=end_time - start_time,
                                                  tokens=tokens, attempts=self._max_attempts)

                    self._code = extract_markdown_code_string(debug_message.code)
                    self._attempts += 1
                    first_chat = False
                else:
                    end_time = time.perf_counter()
                    return TestExecCodeResult(final_function=self._code, passed=True, time= end_time - start_time, tokens = tokens, attempts=self._attempts)

            # Max attempts number reached
            end_time = time.perf_counter()
            return TestExecCodeResult(final_function=self._code, passed=False, time= end_time - start_time, tokens = tokens, attempts=self._max_attempts)

