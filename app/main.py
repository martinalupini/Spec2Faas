from pathlib import Path
import asyncio
import logging
import os
from autogen_core import TRACE_LOGGER_NAME
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.Assistant import *

def load_env_variables():

    try:
        # load environment variables from .env file (requires `pip install python-dotenv`)
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"

        load_dotenv(dotenv_path=env_path)

    except ImportError:
        pass


def set_logging_config():
    log_path = os.getenv("LOG_PATH")

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(TRACE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # File's handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Forcing reconfiguration of logging
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler], force=True)

    logging.getLogger("autogen_core").setLevel(logging.DEBUG)
    logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.DEBUG)

    logging.getLogger(TRACE_LOGGER_NAME).propagate = True


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",  # o "gemini-2.0-pro", "gemini-1.5-pro", ecc.
        api_key=os.environ["GEMINI_API_KEY"],  # chiave di Google AI Studio
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # perché il modello non è “OpenAI”, serve descriverne le capacità:
        model_info={
            "family": ModelFamily.GEMINI_2_0_FLASH,
            "function_calling": True,
            "json_output": True,
            "vision": False,
            "structured_output" : True,
        },
    )
    runtime = SingleThreadedAgentRuntime()
    await Assistant.register(runtime, "assistant", lambda: Assistant(model_client=model_client))
    runtime.start()  # Start processing messages in the background.

    response = Message("","request")
    print("Hi! Write here your function to deploy or the specification of the function you want to write.\n")

    while response.type == "request":
        user_input = input()
        response = await runtime.send_message(Message(user_input, type="request"), AgentId("assistant", "default"))
        print(response.content)
    await runtime.stop()  # Stop processing messages in the background.
    await model_client.close()


if __name__ == "__main__":

    load_env_variables()
    set_logging_config()

    try:
        asyncio.run(main())
    finally:
        logging.shutdown()