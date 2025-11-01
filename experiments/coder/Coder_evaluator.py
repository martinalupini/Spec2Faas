import logging
import asyncio
import sys
import tempfile
import sys
import time
from typing import List
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from app.agents.coding_agents.Coder import *
from autogen_core.tools import FunctionTool, Tool
from app.agents.coding_agents.utils.Utils import *
from app.agents.coding_agents.utils.Code_Extractors import *
from app.Utils import *
import pandas as pd
from Metrics import *



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


async def main(llm, client):

    runtime = SingleThreadedAgentRuntime()
    await Coder.register(runtime, "coder", lambda: Coder(llm = llm, model_client=client))

    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    # Creating file to store data
    file_name = "coder_results/"+ llm+"2.parquet"
    columns = [
        'task_id', 'passed', 'generation time', 'tokens',
        'execution time', 'execution time canonical',
        'CC generation', 'CC canonical', 'CoG generation', 'CoG canonical'
    ]
    if os.path.exists(file_name):
        results_df = pd.read_parquet(file_name)
    else:
        results_df = pd.DataFrame(columns=columns)

    runtime.start()  # Start processing messages in the background.

    # Iterating through each row
    for row in df.itertuples(index=False):
        task_id = row.task_id
        entry_point = row.entry_point
        prompt = row.prompt
        test = row.test
        canonical_solution = row.canonical_solution

        # Function already generated in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        print_yellow(task_id)
        response= await runtime.send_message(TestCodeMessage(prompt, entry_point), AgentId("coder", "default"))


        # Generated function execution
        function_code = extract_markdown_code_blocks(response.content)
        function_code_string = function_code[0].code
        result, execution_time_generated = await execute_function(function_code_string, test, entry_point, executor, response.ctx)
        if result.output == "":
            passed = True
        else:
            passed = False

        CC_generated = compute_CC(function_code_string)
        CoG_generated = compute_CoG(function_code_string)


        # Canonical solution execution
        signature = extract_signature(prompt)
        non_indent_code = signature + canonical_solution
        canonical_function_code = fix_indent(non_indent_code)
        result, execution_time_canonical = await execute_function(canonical_function_code, test, entry_point, executor, response.ctx)
        CC_canonical = compute_CC(canonical_function_code)
        CoG_canonical = compute_CoG(canonical_function_code)

        new_data = {
            'task_id': [str(task_id)],
            'passed': [passed],
            'generation time': [response.time],
            'tokens': [response.tokens],
            'execution time': [execution_time_generated],
            'execution time canonical': [execution_time_canonical],
            'CC generation': [CC_generated],
            'CC canonical': [CC_canonical],
            'CoG generation': [CoG_generated],
            'CoG canonical': [CoG_canonical]
        }

        new_row_df = pd.DataFrame(new_data)
        if results_df.empty:
            results_df = new_row_df
        else:
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        results_df['task_id'] = results_df['task_id'].astype('string')
        results_df.to_parquet(file_name, engine='pyarrow')

    await executor.stop()
    await runtime.stop()  # Stop processing messages in the background.





if __name__ == "__main__":

    load_env_variables()
    log_path = "../../output/coder/log"
    set_logging_config(log_path)
    llm = get_config_data("../../config.yaml")
    coder = llm['coder']

    if coder == "gemini-2.5-pro" or coder == "gemini-2.0-flash":
        model_client = OpenAIChatCompletionClient(
            model=coder,
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
        model_client = None

    try:
        asyncio.run(main(coder, model_client))
    finally:
        logging.shutdown()


