import logging
import asyncio
import sys
import paramiko
import tempfile
import time
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from experiments.faas_deployer.Utils import *
from app.agents.coding_agents.TestDesigner import *
from app.agents.coding_agents.Coder import *
from app.agents.coding_agents.TestExecutor import *
from app.agents.coding_agents.EntryPoint import *
from app.agents.coding_agents.Debugger import *
from app.agents.FaaSDeployer import *
from app.agents.Assistant import *
from app.agents.coding_agents.utils.Utils import *
from app.agents.coding_agents.utils.Code_Extractors import *
from app.Utils import *
from experiments.MessageTypesTest import *
import pandas as pd
from experiments.coder.Metrics import *

"""
It executes the function given the tests provided by the dataset
"""
async def execute_function(function: str, test:str, entry_point, executor, ctx):
    # Installing dependencies in container
    dependencies = "```sh\npip install numpy\n```"
    code_block = extract_markdown_code_blocks(dependencies)

    code = function + test + "\n\ncheck(" + entry_point + ")\n"
    invocation_code = CodeBlock(code=code, language='python')
    code_block.append(invocation_code)

    start_time = time.perf_counter()
    result = await executor.execute_code_blocks(code_block, ctx)
    end_time = time.perf_counter()

    return result, end_time - start_time



async def main(config):
    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    attempt = config['experiment_number']

    # Creating files to store data
    file_name = "system_results/"+ "experiment_" + str(attempt) + "/results.parquet"

    df = pd.read_parquet(file_name)

    humaneval = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    ################################# STARTING TESTS ########################################
    # Iterating through each row
    for index, row in df.iterrows():
        task_id = row.task_id

        he_row = humaneval[humaneval['task_id'] == task_id]
        canonical_tests = he_row['test'].values[0]
        canonical_entry_point = he_row['entry_point'].values[0]

        print_yellow(task_id)

        ######################## TESTING IF ORIGINAL FUNCTION IS CORRECT ######################################

        result, execution_time_generated = await execute_function(row.original_function, canonical_tests, canonical_entry_point,
                                                                  executor, CancellationToken())
        if "Error" in result.output:
            original_function_correct = False
        else:
            original_function_correct = True

        ######################## TESTING IF FINAL FUNCTION IS CORRECT ######################################
        result, execution_time_generated = await execute_function(row.final_function, canonical_tests, canonical_entry_point,
                                                                  executor, CancellationToken())
        if "Error" in result.output:
            final_function_correct = False
        else:
            final_function_correct = True

        df.at[index, 'original_function_correct'] = original_function_correct
        df.at[index, 'final_function_correct'] = final_function_correct

    df.to_parquet(file_name, index=False)

    await executor.stop()






if __name__ == "__main__":

    load_env_variables()
    config = get_config_data_full("../config_test.yaml")


    try:
        asyncio.run(main(config))
    finally:
        logging.shutdown()