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
    def __init__(self, llm:str, model_client: ChatCompletionClient, code_executor: CodeExecutor, server = None) -> None:
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
        self._server = server
        print_green(f"Hi I'm the test executor and I use {self._llm}.")

    @message_handler
    async def handle_execute_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message from {message.sender}")

        # Updating data strucures
        if self._tests == "":
            self._tests = message.tests
        if self._code == "":
            self._code = message.code

        # Only when both Coder and TestDesigner have contacted it it can start code execution
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

                # Extract code block from the response
                code_blocks = extract_markdown_code_blocks(response.content)
                if code_blocks:
                    result = await self._code_executor.execute_code_blocks(
                        code_blocks, cancellation_token=ctx.cancellation_token
                    )
                    if result.output == "":
                        print_yellow(f"\n{'-' * 130}\nExecutor:\nThe function passes all the tests.\n{'-' * 130}")
                        if os.getenv("UI") == "True":
                            self._server.send_chunk("The function passes all the tests.", "test_executor")
                    else:
                        print_yellow(f"\n{'-' * 130}\nExecutor:\n{result.output}\n{'-' * 130}")

                    # If there is an AssertionError the function does not pass all the tests
                    if "Error" in result.output:
                        if os.getenv("UI") == "True":
                            self._server.send_chunk("The function fails the tests. Starting to debug...", "test_executor")
                        debug_message = await self._runtime.send_message(
                            DebugMessage(message.specification, message.code, result.output),
                            AgentId("debugger", "default"))

                        # The executor will start from the new code given by the debugger
                        self._code = debug_message.code
                        # Increasing attempts counter
                        self._attempts += 1
                    else:
                        # The function has been corrected and can be returned
                        return Message(self._code, "test_executor_response")
            # Max attempts number reached
            if os.getenv("UI") == "True":
                self._server.send_chunk("Maximum number of debugging attempts reached.", "test_executor")
            return Message("FAIL", "test_executor_response")

        # This is returned the first time the executor receives a message from Coder or TestDesigner
        return Message("Still waiting", "test_executor_response")

    @message_handler
    async def handle_test_execute_code_message(self, message: TestExecCodeMessage,
                                               ctx: MessageContext) -> TestExecCodeResult:
        print_green(f"{self.id.type} received message.")
        self._tests = message.tests
        self._code = message.code
        entry_point = message.function_signature
        tokens = 0
        first_chat = True
        self._attempts = 0
        start_time = time.perf_counter()
        first_error = ""

        while self._attempts < self._max_attempts:

            dependencies = "```sh\npip install numpy\n```"
            code_blocks = extract_markdown_code_blocks(dependencies)

            code = self._code + self._tests + "\n\ncheck(" + entry_point + ")\n"
            invocation_code = CodeBlock(code=code, language='python')
            code_blocks.append(invocation_code)

            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=ctx.cancellation_token
            )

            if "Error" in result.output:
                print_blue(f"\n{'-' * 130}\nExecutor:\n{result.output}\n{'-' * 130}")
                if self._attempts == 0:
                    first_error = result.output

                debug_message = await self._runtime.send_message(
                    TestDebugMessage(message.specification, message.code, result.output, first_chat),
                    AgentId("debugger", "default"))

                tokens += debug_message.tokens

                if debug_message.code == "":
                    end_time = time.perf_counter()
                    return TestExecCodeResult(final_function=self._code, passed=False, time=end_time - start_time,
                                              tokens=tokens, attempts=self._max_attempts, first_error=first_error, last_error=result.output)

                self._code = extract_markdown_code_string(debug_message.code)
                self._attempts += 1
                first_chat = False
            else:
                end_time = time.perf_counter()
                return TestExecCodeResult(final_function=self._code, passed=True, time=end_time - start_time,
                                          tokens=tokens, attempts=self._attempts, first_error=first_error, last_error=result.output)

        # Max attempts number reached
        end_time = time.perf_counter()
        return TestExecCodeResult(final_function=self._code, passed=False, time=end_time - start_time, tokens=tokens,
                                  attempts=self._max_attempts, first_error=first_error, last_error=result.output)


    @message_handler
    async def handle_test_system_execute_code_message(self, message: TestSystemMessage, ctx: MessageContext) -> TestSystemMessage:
        print_green(f"{self.id.type} received message from {message.sender}")

        # Resetting data structures when it's a new chat
        if message.new_chat:
            self._code = ""
            self._tests = ""
            self._attempts = 0

        # Updating data structures
        if self._tests == "":
            self._tests = message.tests
        if self._code == "":
            self._code = message.code

        # Only when both Coder and TestDesigner have contacted it it can start code execution
        if self._tests != "" and self._code != "":
            print_purple(str(message))
            total_tokens = message.tokens
            total_time = message.time
            self._attempts = 0
            tokens_executor = 0
            time_executor = 0
            tokens_debugger = 0
            time_debugger = 0
            first_chat = True
            start_time = time.perf_counter()
            first_error = ""

            # Loop until max attempts are reached
            while self._attempts < self._max_attempts:
                # Prepare input to the chat completion model.
                prompt = "Function code: " + self._code + "\nFunction tests: " + self._tests
                user_message = UserMessage(content=prompt, source="user")
                response = await self._model_client.create(
                    self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
                )

                usage_metadata = response.usage
                tokens = usage_metadata.prompt_tokens + usage_metadata.completion_tokens
                # Updating token of executor (sum of the tokens of each execution attempts)
                tokens_executor += tokens

                assert isinstance(response.content, str)
                # Extracting code block
                code_blocks = extract_markdown_code_blocks(response.content)
                if code_blocks:
                    result = await self._code_executor.execute_code_blocks(
                        code_blocks, cancellation_token=ctx.cancellation_token
                    )
                    end_time = time.perf_counter()
                    # Updating time of executor (sum of the time of each execution attempts)
                    time_executor += end_time - start_time
                    # If the execution fails contacting the Debugger
                    if "Error" in result.output:
                        if self._attempts == 0:
                            first_error = result.output
                        start_time_debug = time.perf_counter()
                        debug_message = await self._runtime.send_message(
                            TestDebugMessage(message.prompt, message.code, result.output, first_chat),
                            AgentId("debugger", "default"))
                        end_time_debug = time.perf_counter()

                        # Updating time and tokens of debugger as the sum of each debugging attempt
                        tokens_debugger += end_time_debug - start_time_debug
                        time_debugger += end_time_debug - start_time_debug

                        # If the Debugger returns an empty string the debugging process was interrupted for an exception
                        if debug_message.code == "":
                            # Updating time and tokens
                            total_time['test_executor'] = time_executor
                            total_tokens['test_executor'] = tokens_executor
                            total_tokens['debugger'] = tokens_debugger
                            total_time['debugger'] = time_debugger
                            # Original function is not correct and can't be generated
                            return TestSystemMessage(tokens = total_tokens, time=total_time, messages= message.messages +1 + self._attempts, prompt = message.prompt, signature = message.signature, original_func = message.original_func, code = message.code,
                                                     tests = message.tests, tests_str = message.tests_str,
                                                     attempts = self._attempts, type = "test_executor_response", first_error=first_error, last_error=result.output)

                        # The executor will start from the new code given by the debugger
                        self._code = debug_message.code
                        # Increasing attempts counter
                        self._attempts += 1
                        # Is no longer a first chat for the debugger
                        first_chat = False
                    else:
                        # Updating time and tokens
                        total_time['test_executor'] = time_executor
                        total_tokens['test_executor'] = tokens_executor
                        total_tokens['debugger'] = tokens_debugger
                        total_time['debugger'] = time_debugger

                        # Extracting the function code string
                        function_code = extract_markdown_code_blocks(self._code)
                        if function_code:
                            final_function = function_code[0].code
                        else:
                            final_function = self._code

                        # Function has been corrected
                        return TestSystemMessage(tokens=total_tokens, time=total_time, messages= message.messages +1 + self._attempts, prompt=message.prompt,
                                                 signature=message.signature, original_func=message.original_func,
                                                 code=message.code, tests=message.tests, tests_str=message.tests_str,
                                                 final_func = final_function, code_final_func=self._code,
                                                 attempts=self._attempts, generated = True, type="test_executor_response", first_error=first_error, last_error=result.output)

            # Max attempts number reached, function is not correct
            # Updating time and tokens
            total_time['test_executor'] = time_executor
            total_tokens['test_executor'] = tokens_executor
            total_tokens['debugger'] = tokens_debugger
            total_time['debugger'] = time_debugger
            return TestSystemMessage(tokens=total_tokens, time=total_time, messages= message.messages +1 + self._attempts, prompt=message.prompt,
                                     signature=message.signature, original_func=message.original_func,
                                     code=message.code, tests=message.tests, tests_str=message.tests_str,
                                     attempts=self._max_attempts, type="test_executor_response", first_error=first_error, last_error=result.output)

        # returns when the coder sends the message
        return message