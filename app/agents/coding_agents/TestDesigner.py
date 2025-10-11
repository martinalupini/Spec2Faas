from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *

class TestDesigner(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Skilled test designer")
        self._system_messages = [SystemMessage(
            #prompt from https://github.com/huangd1999/AgentCoder/blob/main/prompts/zero_shot_test_designer_humaneval_prompt.txt
            content="You are a very skilled test designer."
                    "**Role**: As a tester, your task is to create comprehensive test cases for the function specification provided as input. "
                    "These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability."
                    "**1. Basic Test Cases**:"
                    "- **Objective**: To verify the fundamental functionality of the `has_close_elements` function under normal conditions."
                    "**2. Edge Test Cases**:"
                    "- **Objective**: To evaluate the function's behavior under extreme or unusual conditions."
                    "**3. Large Scale Test Cases**:"
                    "- **Objective**: To assess the function’s performance and scalability with large data samples."
                    "**Instructions**:"
                    "- Implement a comprehensive set of test cases following the guidelines above."
                    "- Ensure each test case is well-documented with comments explaining the scenario it covers."
                    "- Pay special attention to edge cases as they often reveal hidden bugs."
                    "- For large-scale tests, focus on the function's efficiency and performance under heavy loads."
                    "- The format of test cases should be:"
                    "```python"
                    "assert function_name(input) == expected_output, 'Test Case Description'"
                    "```"
                    "MAKE SURE THE OUTPUT FOLLOWS THIS FORMAT"
                    "MAKE SURE TO USE THE FUNCTION SIGNATURE PROVIDED."
                    "DON'T TRY TO IMPLEMENT THE FUNCTION BUT ONLY THE TESTS"
                    "Return ONLY the test cases in the format specified above, don't return the function signature received in input."

        )]
        self._model_client = model_client

    @message_handler
    async def handle_generate_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Creating now a test suite for the function.")

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
        user_message = UserMessage(content=prompt, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        print_purple(response.content)
        return_message = await self._runtime.send_message(CodeMessage(message.specification, message.function_signature, "", response.content, self.id.type), AgentId("test_executor", "default"))
        return return_message