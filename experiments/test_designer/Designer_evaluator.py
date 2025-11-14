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

from app.agents.coding_agents.TestDesigner import *
from app.agents.coding_agents.utils.Utils import *
from app.agents.coding_agents.utils.Code_Extractors import *
from app.Utils import *
from experiments.MessageTypesTest import *
import pandas as pd



async def execute_function(function: str, test:str, entry_point, executor, ctx):
    # Installing dependencies in container
    dependencies = "```sh\npip install numpy\npip install coverage```"
    code_block = extract_markdown_code_blocks(dependencies)

    code = "import coverage\n\ncov = coverage.Coverage()\n\ncov.start()\n\n" + function + test +"\n\ncov.stop()\n\ncov.report()"
    invocation_code = CodeBlock(code=code, language='python')
    code_block.append(invocation_code)

    start_time = time.perf_counter()
    result = await executor.execute_code_blocks(code_block, ctx)
    end_time = time.perf_counter()

    return result, end_time - start_time


async def main(llm, client, system_prompt):

    runtime = SingleThreadedAgentRuntime()
    await TestDesigner.register(runtime, "test_designer", lambda: TestDesigner(llm = llm, model_client=client))

    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    # Creating file to store data
    if system_prompt:
        file_name = "designer_results/"+ llm+".parquet"
    else:
        file_name = "designer_results/"+ llm+"_no_prompt.parquet"
    columns = [
        'task_id', 'passed', 'generation_time', 'tokens',
        'execution_time', 'tests', 'coverage'
    ]
    if os.path.exists(file_name):
        results_df = pd.read_parquet(file_name)
        if 'coverage' not in results_df.columns:
            results_df['coverage'] = pd.Series(dtype='Int64')
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
        canonical_code = prompt + canonical_solution

        # Function already generated in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        print_yellow(task_id)
        response= await runtime.send_message(TestCodeMessage(prompt, entry_point, system_prompt), AgentId("test_designer", "default"))


        # Generated function execution
        test_code = extract_markdown_code_blocks(response.content)
        if test_code:
            test_code_string = test_code[0].code
            result, execution_time_generated = await execute_function(canonical_code, test_code_string, entry_point, executor, response.ctx)
            if "AssertionError" in result.output:
                passed = False
                coverage = 0
            else:
                passed = True
                match = re.search(r'(\d+)\s*%', result.output)
                if match:
                    coverage = int(match.group(1))
                else:
                    coverage = 0
        else:
            passed = False


        if "AssertionError" in result.output:
            print_purple(f"\n{'-' * 130}\nExecutor:\n{result.output}\n{'-' * 130}")


        new_data = {
            'task_id': [str(task_id)],
            'passed': [passed],
            'generation_time': [response.time],
            'tokens': [response.tokens],
            'execution_time': [execution_time_generated],
            'tests': [test_code_string],
            'coverage': [coverage]
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
    log_path = "../../output/designer/log"
    set_logging_config(log_path)
    llm = get_config_data("../config_test.yaml")
    designer = llm['test_designer']
    if llm['coder_prompt'] == "Yes":
        prompt = True
    else:
        prompt = False

    if designer == "gemini-2.5-pro" or designer == "gemini-2.0-flash":
        model_client = OpenAIChatCompletionClient(
            model=designer,
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
            model=designer,
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
        asyncio.run(main(designer, model_client, prompt))
    finally:
        logging.shutdown()


