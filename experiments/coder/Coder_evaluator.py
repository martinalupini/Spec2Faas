import logging
import asyncio
import sys
import tempfile
import sys
from typing import List
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from app.agents.coding_agents.Coder import *
from autogen_core.tools import FunctionTool, Tool
from app.agents.coding_agents.utils.Utils import *
from app.Utils import *
import pandas as pd

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
        canonical_solution = row.canonical_solution

        # Function already generated in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        response = await runtime.send_message(TestCodeMessage(prompt, entry_point), AgentId("coder", "default"))
        print_yellow(response.content + "\n" + str(response.tokens) + "\n" + str(response.time))

        new_data = {
            'task_id': [task_id],
            'passed': [False],
            'generation time': [response.time],
            'tokens': [response.tokens],
            'execution time': ["NA"],
            'execution time canonical': ["NA"],
            'CC generation': ["NA"],
            'CC canonical': ["NA"]
        }

        new_row_df = pd.DataFrame(new_data)
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        results_df.to_parquet(file_name, engine='pyarrow')


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


