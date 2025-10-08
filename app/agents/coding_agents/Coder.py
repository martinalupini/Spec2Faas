from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *

class Coder(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled software programmer")
        self._system_messages = [SystemMessage(
            # Prompt from https://github.com/huangd1999/AgentCoder/blob/main/prompts/zero_shot_humaneval_prompt.txt
            content="You are a very skilled software programmer."
                    "**Task**: As a programmer, you are required to code a function that adheres to the function specification."
                    "Make sure to use the function signature provided."
                    "Return only the code of the function in the format "
                    #"Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language."
                    "**Code Formatting**: Please write code in this format:"
                    "```python"
                    "[Code]"
                    "```"
                    "**Instructions**:"
                    "1. **Understand and Clarify**: Make sure you understand the task."
                    "2. **Algorithm/Method Selection**: Decide on the most efficient way."
                    #"3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode."
                    #"4. **Code Generation**: Translate your pseudocode into executable Python code. "
                    "3. **Correctness and Completeness: Make sure your code is correct and complete."
                    "4. RETURN ONLY THE CODE OF THE FUNCTION IN THE SPECIFIED FORMAT"
        )]
        self._model_client = model_client

    @message_handler
    async def handle_generate_code_message(self, message: CodeMessage, ctx: MessageContext) -> None:
        print_green(f"{self.id.type} received message. Staring to generate code.")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
        user_message = UserMessage(content=prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        print_purple(response.content)