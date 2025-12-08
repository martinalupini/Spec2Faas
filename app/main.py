import logging
import asyncio
import os
import tempfile
import sys
from typing import List
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
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

from server import Server


async def main(llm, server, user_text):
    models = {}

    for llm_name in llm.keys():
        system_component = llm[llm_name]
        if system_component == "gemini-2.5-pro" or system_component == "gemini-2.0-flash":
            llm_model = OpenAIChatCompletionClient(
                model=system_component,
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                max_retries=10,
                model_info={
                    "family": ModelFamily.GEMINI_2_0_FLASH,
                    "function_calling": True,
                    "json_output": True,
                    "vision": False,
                    "structured_output": True,
                },
            )
        else:
            llm_model = OllamaChatCompletionClient(
                model=system_component,
                host="160.80.97.151:11434",
                model_info={
                    "family": ModelFamily.UNKNOWN,
                    "function_calling": True,
                    "json_output": True,
                    "vision": False,
                    "structured_output": True,
                }
            )
        models[llm_name] = llm_model

    work_dir = tempfile.mkdtemp()
    runtime = SingleThreadedAgentRuntime()
    await Assistant.register(runtime, "assistant", lambda: Assistant(llm = llm['assistant'],model_client=models['assistant'], server = server))
    await EntryPoint.register(runtime, "entry_point", lambda: EntryPoint(llm = llm['entry_point'], model_client=models['entry_point'], server = server))
    await Coder.register(runtime, "coder", lambda: Coder(llm = llm['coder'], model_client=models['coder'], server = server))
    await TestDesigner.register(runtime, "test_designer", lambda: TestDesigner(llm = llm['test_designer'], model_client=models['test_designer'], server = server))
    await Debugger.register(runtime, "debugger", lambda: Debugger(llm = llm['debugger'], model_client=models['debugger'], server = server))
    # creating the tools for the FaaS deployer
    tools: List[Tool] = [FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge and deploy the function on Serveledge.")]
    await FaasDeployer.register(runtime, "faas_deployer", lambda: FaasDeployer(llm = llm['faas_deployer'], model_client=models['faas_deployer'], tool_schema=tools, server = server))
    # Registering the Test Executor
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    #This method sets the working environment variables, connects to Docker and starts the code executor. If no working directory was provided to the code executor, it creates a temporary directory and sets it as the code executor working directory.
    #https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.code_executors.docker.html#autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor
    await executor.start()
    await TestExecutor.register(runtime, "test_executor", lambda: TestExecutor(llm = llm['test_executor'], model_client = models['test_executor'], code_executor = executor, server = server))


    runtime.start()  # Start processing messages in the background.

    if os.getenv("UI") == "False":
        response = Message("","request")
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
                dialogue(
                    "The input is not clear. Please provide more information about the function you want to deploy.",
                    "Assistant")

    else:
        response = await runtime.send_message(Message(user_text, type="request"), AgentId("assistant", "default"))
        if response.type == "request":
            dialogue("The input is not clear. Please provide more information about the function you want to deploy.", "Assistant")
            await executor.stop()
            await runtime.stop()  # Stop processing messages in the background.
            for model in models.values():
                await model.close()
            exit()

    await executor.stop()
    await runtime.stop()  # Stop processing messages in the background.
    for model in models.values():
        await model.close()


if __name__ == "__main__":

    os.environ['UI'] = 'False'
    load_env_variables()
    log_path = os.getenv("LOG_PATH")
    set_logging_config(log_path)
    llm = get_config_data("../config.yaml")

    try:
        asyncio.run(main(llm, None, ""))
    finally:
        logging.shutdown()


