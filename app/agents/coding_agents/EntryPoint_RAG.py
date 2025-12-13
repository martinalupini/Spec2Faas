from unittest import TestResult
import os
from autogen_core import MessageContext, RoutedAgent, message_handler, AgentId
from autogen_core.models import UserMessage, ChatCompletionClient, SystemMessage

from experiments.MessageTypesTest import TestMessageResult
from .messages.MessagesTypes import *
from experiments.MessageTypesTest import *
from .utils.Utils import *
import time


class EntryPoint_RAG(RoutedAgent):
    def __init__(self, llm: str,  model_client: ChatCompletionClient, server = None, memory = None) -> None:
        super().__init__("The entry point of the coding.")
        self._system_messages = [SystemMessage(
            content="Given the function specification and the memory context create a signature for the function."
                    "Do not implement the function."
                    "Return only the signature of the function as output."
        )]
        self._model_client = model_client
        self._llm = llm
        self._server = server
        self._memory = memory
        print_green(f"Hi I'm the entry point of the coding system and I use {self._llm}.")

    @message_handler
    async def handle_assistant_message(self, message: Message, ctx: MessageContext) -> Message:
        # If the message type is "test_executor_response" then the entire coding process is concluded
        if message.type == "test_executor_response":
            print_green(f"{self.id.type} received message. Communicating final outcome.")
            return message

        # If the message is not of this type then is the Assistant that wants to initiate the coding process
        print_green(f"{self.id.type} received message. Activating Coder and Test Designer.")

        search_query = message.content
        rag_context = ""

        retrieved_docs = await self._memory.query(search_query)
        print(retrieved_docs)

        if retrieved_docs:
            retrieved_content = "\n\n".join([doc.content for doc in retrieved_docs.results])
            rag_context = (
                "\n\n<CONTEXT FROM MEMORY>\n"
                "The following useful pseudocode was retrieved from memory:\n"
                f"{retrieved_content}"
                "\n</CONTEXT FROM MEMORY>\n"
            )

            memory_content = "The following useful code pseudocode were retrieved from memory:\n" + retrieved_content + "\n\n"

        else:
            memory_content = "No content was retrieved from memory.\n\n"

        rag_message = [SystemMessage(content=rag_context)]
        system_message_list = self._system_messages + rag_message
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            system_message_list + [user_message], cancellation_token=ctx.cancellation_token
        )
        dialogue(memory_content + "\n\n Asking to generate code and test for the function " + response.content, "Entry Point")
        if os.getenv("UI") == "True":
            self._server.send_chunk(memory_content + "\n\n Asking to generate code and test for the function " + response.content, "entry_point")

        # After generating the signature the EntryPoint send a message both to the Coder and the TestDesigner
        await self._runtime.send_message(CodeMessage(message.content, response.content, "", "", self.id.type, rag_context), AgentId("coder", "default"))
        return_message = await self._runtime.send_message(CodeMessage(message.content, response.content, "", "", self.id.type, rag_context), AgentId("test_designer", "default"))

        return return_message
