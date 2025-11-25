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


def modify_tests(test: str) -> str:
    line_to_insert = '        out = candidate(*inp)\n        if out != exp:\n            print("input: "+ str(inp))\n            print("expected output: "+ str(exp))\n            print("actual output: "+ str(out))'

    lines = test.split('\n')

    target_index = -1
    assertion_index = -1

    # Trova gli indici delle righe di interesse
    for i, line in enumerate(lines):
        if line.strip().startswith('for i, (inp, exp) in enumerate(zip(inputs, results)):'):
            target_index = i
        elif line.strip().startswith('assertion('):
            assertion_index = i

    # Inserisci il blocco di debug dopo la riga del ciclo for
    if target_index != -1:
        lines.insert(target_index + 1, line_to_insert)
        # Se la riga assertion era successiva al punto di inserimento, aggiorna il suo indice
        if assertion_index > target_index:
            assertion_index += 1

    # Indenta la riga dell'assertion con 4 spazi (lo stesso livello dell'if aggiunto)
    if assertion_index != -1:
        lines[assertion_index] = '        ' + lines[assertion_index].lstrip()

    new_string = '\n'.join(lines)
    return new_string



async def main(llm, client, system_prompt, debugger, client_debugger):

    runtime = SingleThreadedAgentRuntime()

    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    await Coder.register(runtime, "coder", lambda: Coder(llm = llm, model_client=client))
    await Debugger.register(runtime, "debugger", lambda: Debugger(llm=debugger, model_client=client_debugger))
    await TestExecutor.register(runtime, "test_executor", lambda: TestExecutor(llm="", model_client=client, code_executor=executor))

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    # Creating file to store data
    if system_prompt:
        file_name = "debugger_results/"+ llm + "_" + debugger +".parquet"
    else:
        file_name = "debugger_results/"+  llm + "_" + debugger +"_no_prompt.parquet"
    columns = [
        'task_id', 'passed', 'passed_after_debugging','debugging_tokens', 'total_tokens',
        'generation_time', 'debugging_time', 'attempts',
        'CC_generation', 'CC_canonical', 'CoG_generation', 'CoG_canonical', 'Original_function', 'Debugged_function', 'CC_debugged', 'CoG_debugged', 'first_error', 'last_error'
    ]
    if os.path.exists(file_name):
        results_df = pd.read_parquet(file_name)
    else:
        results_df = pd.DataFrame(columns=columns)

    # Retrieving data from the coder file
    coder_path = "../coder/coder_results/" + coder + ".parquet"
    coder_df = pd.read_parquet(coder_path)

    runtime.start()  # Start processing messages in the background.

    # Iterating through each row
    for row, row_coder_df in zip(df.itertuples(index=False), coder_df.itertuples(index=False)):
        task_id = row.task_id

        # Function already generated in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        entry_point = row.entry_point
        prompt = row.prompt
        test = modify_tests(row.test)

        print_yellow(task_id)

        function_code_string = row_coder_df.function
        CC_generated = row_coder_df.CC_generation
        CoG_generated = row_coder_df.CoG_generation
        CC_canonical = row_coder_df.CC_canonical
        CoG_canonical = row_coder_df.CoG_canonical

        generation_time = row_coder_df.generation_time

        if row_coder_df.passed:
            passed_after_debugging = True
            debugging_tokens = 0
            total_tokens = row_coder_df.tokens
            debugging_time = 0
            debugging_attempts = 0
            CC_debugged = 0
            CoG_debugged = 0
            final_function = function_code_string
            first_error = ""
            last_error = ""

        else:

            response_debug= await runtime.send_message(TestExecCodeMessage(prompt, entry_point, function_code_string, test), AgentId("test_executor", "default"))


            CC_debugged = compute_CC(response_debug.final_function)
            CoG_debugged = compute_CoG(response_debug.final_function)

            passed_after_debugging = response_debug.passed
            debugging_tokens = response_debug.tokens
            total_tokens = row_coder_df.tokens + response_debug.tokens
            debugging_time = response_debug.time
            debugging_attempts = response_debug.attempts
            final_function = response_debug.final_function

            first_error = response_debug.first_error
            last_error = response_debug.last_error



        new_data = {
            'task_id': [str(task_id)],
            'passed': [row_coder_df.passed],
            'passed_after_debugging': [passed_after_debugging],
            'debugging_tokens': [debugging_tokens],
            'total_tokens': [total_tokens],
            'generation_time': [generation_time],
            'debugging_time': [debugging_time],
            'attempts': [debugging_attempts],
            'CC_generation': [CC_generated],
            'CC_canonical': [CC_canonical],
            'CoG_generation': [CoG_generated],
            'CoG_canonical': [CoG_canonical],
            'Original_function': [function_code_string],
            'Debugged_function': [final_function],
            'CC_debugged': [CC_debugged],
            'CoG_debugged': [CoG_debugged],
            'first_error': [first_error],
            'last_error': [last_error]
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
    await model_client.close()



if __name__ == "__main__":

    load_env_variables()
    log_path = "../../output/debugger/log"
    set_logging_config(log_path)
    llm = get_config_data("../config_test.yaml")
    coder = llm['coder']
    debugger = llm['debugger']
    if llm['coder_prompt'] == "Yes":
        prompt = True
    else:
        prompt = False

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
        model_client = OllamaChatCompletionClient(
            model=coder,
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
        asyncio.run(main(coder, model_client, prompt, debugger, model_client_debugger))
    finally:
        logging.shutdown()


