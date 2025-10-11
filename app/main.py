from pathlib import Path
import asyncio
import logging
from typing import List
import os
import sys
import tempfile
from autogen_core import TRACE_LOGGER_NAME
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from agents.Assistant import *
from agents.coding_agents.Coder import *
from agents.coding_agents.EntryPoint import *
from agents.coding_agents.TestDesigner import *
from agents.coding_agents.TestExecutor import *
from agents.coding_agents.Debugger import *
from agents.FaaSDeployer import *
from autogen_core.tools import FunctionTool, Tool
from agents.coding_agents.utils.Utils import *

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
    work_dir = tempfile.mkdtemp()
    runtime = SingleThreadedAgentRuntime()
    await Assistant.register(runtime, "assistant", lambda: Assistant(model_client=model_client))
    await EntryPoint.register(runtime, "entry_point", lambda: EntryPoint(model_client=model_client))
    await Coder.register(runtime, "coder", lambda: Coder(model_client=model_client))
    await TestDesigner.register(runtime, "test_designer", lambda: TestDesigner(model_client=model_client))
    await Debugger.register(runtime, "debugger", lambda: Debugger(model_client=model_client))
    # creating the tools for the FaaS deployer
    tools: List[Tool] = [FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge.")]
    await FaasDeployer.register(runtime, "faas_deployer", lambda: FaasDeployer(model_client=model_client, tool_schema=tools))
    # Registering the Test Executor
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    #This method sets the working environment variables, connects to Docker and starts the code executor. If no working directory was provided to the code executor, it creates a temporary directory and sets it as the code executor working directory.
    #https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.code_executors.docker.html#autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor
    await executor.start()
    await TestExecutor.register(runtime, "test_executor", lambda: TestExecutor(model_client,executor))


    runtime.start()  # Start processing messages in the background.

    response = Message("","request")
    print_blue("Hi! Write here your function to deploy or the specification of the function you want to write.\n"
          "Press ENTER two times to continue.")
    input_try = 0

    while response.type == "request":
        lines = []
        for line in sys.stdin:
            if line.strip() == "":
                break
            lines.append(line)

        user_input = "".join(lines)
        response = await runtime.send_message(Message(user_input, type="request"), AgentId("assistant", "default"))
        if response.type == "request":
            print_blue("The input is not clear. Please provide more information about the function you want to deploy.")

    print_green(response.content)
    await runtime.stop()  # Stop processing messages in the background.
    await model_client.close()


if __name__ == "__main__":

    load_env_variables()
    set_logging_config()

    try:
        asyncio.run(main())
    finally:
        logging.shutdown()


