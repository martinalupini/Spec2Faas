import logging
import asyncio
import sys
import tempfile
import sys
import time
from typing import List
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
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


async def main(llm):

    runtime = SingleThreadedAgentRuntime()
    await Coder.register(runtime, "coder", lambda: Coder(llm = llm['coder']))

    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")
    print(df.head())

    # Creating file to store data
    file_name = "coder_results/"+ llm['coder']+".parquet"
    columns = [
        'task_id', 'passed', 'generation time', 'tokens',
        'execution time', 'execution time canonical',
        'CC generation', 'CC canonical'
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

        # Canonical solution execution
        signature = extract_signature(prompt)
        non_indent_code = signature + canonical_solution
        canonical_function_code = fix_indent(non_indent_code)
        result, execution_time_canonical = await execute_function(canonical_function_code, test, entry_point, executor, response.ctx)
        CC_canonical = compute_CC(canonical_function_code)

        """
        new_data = {
            'task_id': [task_id],
            'passed': [passed],
            'generation time': [response.time],
            'tokens': [response.tokens],
            'execution time': [execution_time_generated],
            'execution time canonical': [execution_time_canonical],
            'CC generation': [CC_generated],
            'CC canonical': [CC_canonical]
        }

        new_row_df = pd.DataFrame(new_data)
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        results_df.to_parquet(file_name, engine='pyarrow')
                """

    await executor.stop()
    await runtime.stop()  # Stop processing messages in the background.





if __name__ == "__main__":

    log_path = "../../output/coder/log"
    set_logging_config(log_path)
    llm = get_config_data("../../config.yaml")
    results_path = "./coder_results/" + llm['coder']
    file_name = "coder_results/" + llm['coder'] + ".parquet"
    #df = pd.read_parquet(file_name)
    #print(df.head())

    try:
        asyncio.run(main(llm))
    finally:
        logging.shutdown()


