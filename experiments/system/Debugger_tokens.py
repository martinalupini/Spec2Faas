import logging
import asyncio
import sys
import tempfile
import time
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from experiments.coder.Metrics import *
from app.agents.coding_agents.Coder import *
from app.agents.coding_agents.Debugger import *
from app.agents.coding_agents.TestExecutor import *
from app.agents.coding_agents.utils.Utils import *
from app.agents.coding_agents.utils.Code_Extractors import *
from app.Utils import *
import pandas as pd
from experiments.coder.Metrics import *




async def main(client_executor, debugger, client_debugger, attempt):

    runtime = SingleThreadedAgentRuntime()

    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    await Debugger.register(runtime, "debugger", lambda: Debugger(llm=debugger, model_client=client_debugger))
    await TestExecutor.register(runtime, "test_executor", lambda: TestExecutor(llm="", model_client=client_executor, code_executor=executor))

    file_name = "system_results/" + "experiment_" + str(attempt) + "/results.parquet"

    df = pd.read_parquet(file_name)

    runtime.start()  # Start processing messages in the background.

    # Iterating through each row
    for index, row in df.iterrows():
        task_id = row.task_id

        print_yellow(task_id)

        if row.debugged:

            response_debug= await runtime.send_message(TestExecCodeMessage(row.prompt, row.signature, row.original_function, row.tests), AgentId("test_executor", "default"))

            debugging_tokens = response_debug.tokens

            print(debugging_tokens)
            df.at[index, 'token_debugger'] = debugging_tokens

    df.to_parquet(file_name, index=False)
    await executor.stop()
    await runtime.stop()  # Stop processing messages in the background.
    await model_client.close()



if __name__ == "__main__":

    load_env_variables()
    config = get_config_data_full("../config_test.yaml")
    llm = config['llm']
    executor = llm['test_executor']
    debugger = llm['debugger']
    if llm['coder_prompt'] == "Yes":
        prompt = True
    else:
        prompt = False

    if executor == "gemini-2.5-pro" or executor == "gemini-2.0-flash":
        model_client = OpenAIChatCompletionClient(
            model=executor,
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries = 10,
            model_info={
                "family": ModelFamily.GEMINI_2_0_FLASH,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            },
        )
    else:
        model_client = OllamaChatCompletionClient(
            model=executor,
            host="160.80.97.151:11434",
            model_info={
                "family": ModelFamily.UNKNOWN,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            }
        )

    if debugger == "gemini-2.5-pro" or debugger == "gemini-2.0-flash":
        model_client_debugger = OpenAIChatCompletionClient(
            model=debugger,
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries = 10,
            model_info={
                "family": ModelFamily.GEMINI_2_0_FLASH,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            },
        )
    else:
        model_client_debugger = OllamaChatCompletionClient(
            model=debugger,
            host="160.80.97.151:11434",
            model_info={
                "family": ModelFamily.UNKNOWN,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            }
        )

    try:
        asyncio.run(main(model_client, debugger, model_client_debugger, config['experiment_number']))
    finally:
        logging.shutdown()


