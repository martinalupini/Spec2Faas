from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId, CancellationToken
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from .messages.MessagesTypes import *
from .utils.Utils import *
from .utils.Code_Extractors import *
from experiments.MessageTypesTest import *
import ollama
import time
import os

class Coder_RAG2(RoutedAgent):
    def __init__(self,llm: str, model_client: ChatCompletionClient = None, server = None, memory = None) -> None:
        super().__init__("Skilled software programmer")
        self._system_messages = [SystemMessage(content ="You are a very skilled software programmer."
                    "<TASK>"
                    "As a programmer, you are required to code a function that adheres to the function specification."
                    "You also have access to an external memory from which you can get useful information."
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
                    "4. Make sure your code is correct and complete."
                    "5. If you find useful snippets or functions in memory, use them."
                    "4. RETURN ONLY THE CODE OF THE FUNCTION IN THE SPECIFIED FORMAT"
                    "<I</NSTRUCTIONS>"
        )]
        self._model_client = model_client
        self._llm = llm
        self._text = ""
        self._role = "Software Programmer"
        self._client = None
        self._memory = memory
        self._server = server
        print_green(f"Hi I'm the software programmer and I use {self._llm}.")

    @message_handler
    async def handle_generate_code_message(self, message: CodeMessage, ctx: MessageContext) -> Message:
        print_green(f"{self.id.type} received message. Staring to generate code with {self._llm}.")

        search_query = message.specification

        retrieved_docs = await self._memory.query(search_query)
        print(retrieved_docs)

        if retrieved_docs:
            retrieved_content = "\n\n".join([doc.content for doc in retrieved_docs.results])
            rag_context = (
                "\n\n<CONTEXT FROM MEMORY>\n"
                "The following useful code snippets/functions were retrieved from memory:\n"
                f"{retrieved_content}"
                "\n</CONTEXT FROM MEMORY>\n"
            )

            memory_content ="The following useful code snippets/functions were retrieved from memory:\n" + retrieved_content + "\n\n"
            rag_context = [SystemMessage(content=rag_context)]
            system_message_list =  self._system_messages + rag_context


        else:
            system_message_list = self._system_messages
            memory_content = "No content was retrieved from memory.\n\n"

        # Prepare input to the chat completion model.
        prompt = "Function specification: " + message.specification + "\nFunction signature: " + message.function_signature
        user_message = UserMessage(content=prompt, source="user")

        print(system_message_list)
        response = await self._model_client.create(
           system_message_list + [user_message], cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        dialogue(memory_content + "This is the generated code:\n\n" + response.content, self._role)
        if os.getenv("UI") == "True":
            self._server.send_chunk(memory_content + "This is the generated code:\n\n" + response.content, "coder")

        # Return message to the entry_point
        return_message = await self._runtime.send_message(
            CodeMessage(message.specification, message.function_signature, response.content, "", self.id.type),
            AgentId("test_executor", "default"))
        return return_message