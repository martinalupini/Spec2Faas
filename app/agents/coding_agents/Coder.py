from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *
import ollama

class Coder(RoutedAgent):
    def __init__(self,llm: str, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled software programmer")
        self._system_prompt = """You are a very skilled software programmer.
                    <TASK>
                    As a programmer, you are required to code a function that adheres to the function specification.
                    Make sure to use the function signature provided.
                    <CODE FORMATTING>
                    Please write code in this format:
                    ```python
                    [Code]
                    ```
                    </CODE FORMATTING>
                    <INSTRUCTIONS>
                    1. Make sure you understand the task.
                    2. Decide on the most efficient way to solve the task.
                    3. Use the function signature provided.
                    3. Make sure your code is correct and complete.
                    4. RETURN ONLY THE CODE OF THE FUNCTION IN THE SPECIFIED FORMAT
                    </INSTRUCTIONS>\n"""
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
                    "<I</NSTRUCTIONS>"
        )]
        self._model_client = model_client
        self._llm = llm
        self._text = ""
        self._role = "Software Programmer"
        self._client = None
        print_green(f"Hi I'm the software programmer and I use {self._llm}.")

    @message_handler
    async def handle_generate_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Staring to generate code with {self._llm}.")

        if self._llm == "gemini-2.5-pro" or self._llm == "gemini-2.0-flash":
            # Prepare input to the chat completion model.
            prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
            user_message = UserMessage(content=prompt, source="user")
            response = await self._model_client.create(
                self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
            )

            assert isinstance(response.content, str)
            dialogue(response.content, self._role)
            return_message = await self._runtime.send_message(
                CodeMessage(message.specification, message.function_signature, response.content, "", self.id.type),
                AgentId("test_executor", "default"))
            return return_message

        else:

            if self._client is None:
                self._client = ollama.Client(host='http://160.80.97.151:11434')

            # Prepare input to the chat completion model.
            prompt = "Write a the code given this function specification: " + message.specification + "\n. This is the function signature: " + message.function_signature
            response = self._client.chat(
                model=self._llm,
                messages=[
                    {'role': 'user',
                     'content': self._system_prompt + prompt},
                ]
            )

            dialogue(response['message']['content'], self._role)
            return_message = await self._runtime.send_message(
                CodeMessage(message.specification, message.function_signature, response['message']['content'], "",
                            self.id.type), AgentId("test_executor", "default"))
            return return_message
