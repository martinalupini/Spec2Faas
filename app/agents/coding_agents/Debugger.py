from typing import List
import re
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, LLMMessage, AssistantMessage
from .messages.MessagesTypes import *
from .utils.Utils import *
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import InternalServerError, RateLimitError

class Debugger(RoutedAgent):
    def __init__(self,llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled software debugger")
        self._system_messages = [SystemMessage(content="You are a very skilled software programmer specialized in debugging code."
                    "As a debugger, you are required to correct the code given as input"
                    "<INSTRUCTIONS>"
                    "Step 1: inspect the code in input and the error message."
                   # "Step 2: establish if the error makes sense or the test is wrong. If the test is wrong Respond with 'Test <number> is wrong'"
                    "Step 2: explain the error message and understand why the code is not working"
                    "Step 3: correct the code."
                    "Return ONLY the code of the function in the specified format."
                    "It is important to remember previous corrections of the same function."
                    "</INSTRUCTIONS>"
                    "<OUTPUT FORMAT>"
                    "```python"
                    "[Code]"
                    "```"
                    "</OUTPUT FORMAT>"
        )]
        self._model_client = model_client
        # Adding a memory so that the debugger as an history of what happend
        self._debug_chat: List[LLMMessage] = self._system_messages
        self._counter = 0
        self._llm = llm
        self._role = "Debugger"
        print_green(f"Hi I'm the debugger and I use {self._llm}.")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        # 3. Specifica ESATTAMENTE quali eccezioni devono triggerare un nuovo tentativo
        retry=retry_if_exception_type((InternalServerError, RateLimitError)),
    )
    async def _create_model_response_with_retry(self, messages, cancellation_token):
        """
        Questa funzione isola la chiamata di rete, rendendola sicura per i tentativi.
        """
        return await self._model_client.create(
            messages=messages, cancellation_token=cancellation_token
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),  # Attesa esponenziale tra i tentativi
        stop=stop_after_attempt(5),  # Numero massimo di tentativi
        retry_error_callback=lambda retry_state: print_red(
            f"Attempt {retry_state.attempt_number} failed, new attempt now...")
    )
    @message_handler
    async def handle_debug_code_message(self, message: DebugMessage, ctx: MessageContext) -> DebugMessage:
        print("ao1")
        self._counter += 1
        print_green(f"{self.id.type} received message. Staring to debug code. Attempt {self._counter}")

        # Prepare input to the chat completion model.
        prompt = "This is the function specification: " + message.specification + "\nThis is the function code: " + message.code + "\nAnd this is the error message: " + message.error_message + "\nCan you correct the code?"
        self._debug_chat.append(UserMessage(content=prompt, source="user"))

        # Prepare input to the chat completion model.
        try:
            # 5. Chiama la funzione helper che ora è resiliente
            response = await self._create_model_response_with_retry(
                messages=self._debug_chat,
                cancellation_token=ctx.cancellation_token
            )

            self._debug_chat.append(AssistantMessage(content=response.content, source="assistant"))
            assert isinstance(response.content, str)
            dialogue(response.content, self._role)

            match = re.search(r"(```python\n.*?```)", response.content, re.DOTALL)
            code_block = match.group(1).strip()
            return DebugMessage(message.specification, code_block, "")

        except (InternalServerError, RateLimitError) as e:
            print("ao")
            print_red(f"API didn't work after several attempts: {e}")
            raise e


