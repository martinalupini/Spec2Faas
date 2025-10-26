import logging
import asyncio
import sys
import tempfile
import sys
from typing import List
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
from Utils import *



async def main(llm):

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
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

    model_client_pro = OpenAIChatCompletionClient(
        model="gemini-2.5-pro",
        api_key=os.environ["GEMINI_API_KEY"],  # chiave di Google AI Studio
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # perché il modello non è “OpenAI”, serve descriverne le capacità:
        model_info={
            "family": ModelFamily.GEMINI_2_0_FLASH,
            "function_calling": True,
            "json_output": True,
            "vision": False,
            "structured_output": True,
        },
    )

    work_dir = tempfile.mkdtemp()
    runtime = SingleThreadedAgentRuntime()
    await Assistant.register(runtime, "assistant", lambda: Assistant(llm = llm['assistant'],model_client=model_client))
    await EntryPoint.register(runtime, "entry_point", lambda: EntryPoint(llm = llm['entry_point'], model_client=model_client))
    await Coder.register(runtime, "coder", lambda: Coder(llm = llm['coder'], model_client=model_client))
    await TestDesigner.register(runtime, "test_designer", lambda: TestDesigner(llm = llm['test_designer'], model_client=model_client_pro))
    await Debugger.register(runtime, "debugger", lambda: Debugger(llm = llm['debugger'], model_client=model_client_pro))
    # creating the tools for the FaaS deployer
    tools: List[Tool] = [FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge.")]
    await FaasDeployer.register(runtime, "faas_deployer", lambda: FaasDeployer(llm = llm['faas_deployer'], model_client=model_client, tool_schema=tools))
    # Registering the Test Executor
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    #This method sets the working environment variables, connects to Docker and starts the code executor. If no working directory was provided to the code executor, it creates a temporary directory and sets it as the code executor working directory.
    #https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.code_executors.docker.html#autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor
    await executor.start()
    await TestExecutor.register(runtime, "test_executor", lambda: TestExecutor(llm = llm['test_executor'], model_client = model_client, code_executor = executor))


    runtime.start()  # Start processing messages in the background.

    response = Message("","request")
    #print_blue("Hi! Write here your function to deploy or the specification of the function you want to write.\n"
          #"Press ENTER two times to continue.")

    dialogue("Hi! Write here your function to deploy or the specification of the function you want to write.\n"
          "Press ENTER two times to continue.", "Assistant")

    while response.type == "request":
        lines = []
        for line in sys.stdin:
            if line.strip() == "":
                break
            lines.append(line)

        user_input = "".join(lines)
        response = await runtime.send_message(Message(user_input, type="request"), AgentId("assistant", "default"))
        if response.type == "request":
            dialogue("The input is not clear. Please provide more information about the function you want to deploy.", "Assistant")

    await runtime.stop()  # Stop processing messages in the background.
    await model_client.close()
    await model_client_pro.close()


if __name__ == "__main__":

    load_env_variables()
    set_logging_config()
    llm = get_config_data()

    try:
        asyncio.run(main(llm))
    finally:
        logging.shutdown()


