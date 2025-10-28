from typing import List
import re
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, LLMMessage, AssistantMessage
from .messages.MessagesTypes import *
from .utils.Utils import *

# 1. Importa le eccezioni specifiche di OpenAI e Tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import InternalServerError, RateLimitError  # RateLimitError per errori 429, InternalServerError per 500/503


# Funzione di utilità per il logging dei tentativi
def log_attempt_number(retry_state):
    """Stampa un messaggio di log prima di un nuovo tentativo."""
    print_red(f"Errore API: '{retry_state.outcome.exception()}'. "
              f"Tentativo {retry_state.attempt_number} fallito. "
              f"Attendo {retry_state.next_action.sleep:.2f} secondi prima del prossimo tentativo.")


class Debugger(RoutedAgent):
    def __init__(self, llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled software debugger")
        self._system_messages = [
            SystemMessage(content="You are a very skilled software programmer specialized in debugging code."
                                  "As a debugger, you are required to correct the code given as input"
                                  "<INSTRUCTIONS>"
                                  "Step 1: inspect the code in input and the error message."
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
        self._debug_chat: List[LLMMessage] = self._system_messages
        self._counter = 0
        self._llm = llm
        self._role = "Debugger"
        print_green(f"Hi I'm the debugger and I use {self._llm}.")

    # 2. Applica il decoratore alla nuova funzione helper asincrona
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        # 3. Specifica ESATTAMENTE quali eccezioni devono triggerare un nuovo tentativo
        retry=retry_if_exception_type((InternalServerError, RateLimitError)),
        before_sleep=log_attempt_number
    )
    async def _create_model_response_with_retry(self, messages, cancellation_token):
        """
        Questa funzione isola la chiamata di rete, rendendola sicura per i tentativi.
        """
        return await self._model_client.create(
            messages=messages, cancellation_token=cancellation_token
        )

    # 4. Rimuovi il decoratore @retry da qui
    @message_handler
    async def handle_debug_code_message(self, message: DebugMessage, ctx: MessageContext) -> DebugMessage:
        self._counter += 1
        print_green(f"{self.id.type} received message. Staring to debug code. Attempt {self._counter}")

        # La preparazione del prompt e l'aggiornamento della cronologia avvengono UNA SOLA VOLTA
        prompt = "This is the function specification: " + message.specification + "\nThis is the function code: " + message.code + "\nAnd this is the error message: " + message.error_message + "\nCan you correct the code?"
        self._debug_chat.append(UserMessage(content=prompt, source="user"))

        try:
            # 5. Chiama la funzione helper che ora è resiliente
            response = await self._create_model_response_with_retry(
                messages=self._debug_chat,
                cancellation_token=ctx.cancellation_token
            )

            # Questa parte viene eseguita solo se la chiamata API ha successo
            self._debug_chat.append(AssistantMessage(content=response.content, source="assistant"))
            assert isinstance(response.content, str)
            dialogue(response.content, self._role)

            match = re.search(r"(```python\n.*?```)", response.content, re.DOTALL)
            if not match:
                # Gestisci il caso in cui il modello non restituisce un blocco di codice
                print_red("Warning: No python code block found in the response.")
                return DebugMessage(message.specification, response.content, "No code block found")

            code_block = match.group(1).strip()
            return DebugMessage(message.specification, code_block, "")

        except (InternalServerError, RateLimitError) as e:
            # Questo blocco viene raggiunto solo se Tenacity si arrende dopo tutti i tentativi
            error_message = f"L'API non ha risposto dopo diversi tentativi. Errore finale: {e}"
            print_red(error_message)
            # Rilancia l'eccezione o restituisci un messaggio di errore per informare il chiamante
            raise e
        except Exception as e:
            # Cattura altre eccezioni impreviste
            print_red(f"Si è verificato un errore imprevisto: {e}")
            raise