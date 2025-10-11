from typing import List
import re
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, LLMMessage, AssistantMessage
from .messages.MessagesTypes import *
from .utils.Utils import *

class Debugger(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled software debugger")
        self._system_messages = [SystemMessage(
            content="You are a very skilled software programmer specialized in debugging code."
                    "**Task**: As a debugger, you are required to correct the code given as input"
                    "Step 1: inspect the code in input and the error message."
                    "Step 2: explain the error message and understand why the code is not working"
                    "Step 3: correct the code."
                    "Use a Chain-of-Thought approach to break down the problem and then correct the code."
                    "Make sure to give attention to details such as case punctuation and case when working with strings, floating point approximantion when working with numbers."
                    "Don't be superficial when inspecting the code and really think about what is wrong and try to fix it."
                    "Return only the code of the function in the format format:"
                    "```python"
                    "[Code]"
                    "```"
                    "Example of fixing"
                    "Original code:"
                    "def count_string_occurrences(text, pattern):"
                        "if not pattern:"
                            "return 1"
                        "if len(pattern) > len(text):"
                            "return 0"
                        "count = 0"
                        "for i in range(len(text) - len(pattern) + 1):"
                            "if text[i:i + len(pattern)] == pattern:"
                                "count += 1"
                        "return count"
                    "The error was count_string_occurrences(\"This is a very long string with many occurrences of the word very. Very very very.\", \"very\") returned 4 and not 5."
                    "Fixed code:"
                    "def count_string_occurrences(text, pattern):"
                        "if not pattern:"
                            "return 1"
                        "if len(pattern) > len(text):"
                            "return 0"
                        "count = 0"
                        "for i in range(len(text) - len(pattern) + 1):"
                            "if text[i:i + len(pattern)].lower() == pattern.lower():"
                                "count += 1"
                        "return count"
                    "**Instructions**:"
                    "1. **Understand and Clarify**: Make sure you understand the error and why it's raised."
                    "2. **Completeness**: If you think there are other edge cases not considered make more corrections."
                    "3. **Coherence**: Your tests should be coherent. This is not accectable:"
                    "Test 1. assert is_palindrome(\"A man, a plan, a canal: Panama\") == True"
                    "Test 2. assert is_palindrome(\"No 'x' in Nixon\") == False"
                    "4. RETURN ONLY THE CODE OF THE FUNCTION IN THE SPECIFIED FORMAT"
        )]
        self._model_client = model_client
        # Adding a memory so that the debugger as an history of what happend
        self._debug_chat: List[LLMMessage] = self._system_messages
        self._counter = 0

    @message_handler
    async def handle_debug_code_message(self, message: DebugMessage, ctx: MessageContext) -> DebugMessage:
        self._counter += 1
        print_green(f"{self.id.type} received message. Staring to debug code. Attempt {self._counter}")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction code: " + message.code + " Error message: " + message.error_message
        self._debug_chat.append(UserMessage(content=prompt, source="user"))
        response = await self._model_client.create(
            messages=self._debug_chat, cancellation_token=ctx.cancellation_token
        )
        self._debug_chat.append(AssistantMessage(content=response.content, source="assistant"))

        assert isinstance(response.content, str)
        print_purple(response.content)

        match = re.search(r"(```python\n.*?```)", response.content, re.DOTALL)
        code_block = match.group(1).strip()

        return DebugMessage(message.specification, code_block, "")

