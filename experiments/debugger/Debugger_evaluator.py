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
        lines[assertion_index] = '    ' + lines[assertion_index].lstrip()

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
        file_name = "debugger_results/"+ llm+".parquet"
    else:
        file_name = "debugger_results/"+ llm+"_no_prompt.parquet"
    columns = [
        'task_id', 'passed', 'passed after debugging', 'tokens',
        'generation time', 'debugging time', 'attempts',
        'CC generation', 'CC canonical', 'CoG generation', 'CoG canonical', 'Original function', 'Debugged function'
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
        test = modify_tests(row.test)
        canonical_solution = row.canonical_solution
        debugged_function = ""

        # Function already generated in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        print_yellow(task_id)
        response= await runtime.send_message(TestCodeMessage(prompt, entry_point, system_prompt), AgentId("coder", "default"))

        # Generated function execution
        function_code = extract_markdown_code_blocks(response.content)
        function_code_string = function_code[0].code
        CC_generated = compute_CC(function_code_string)
        CoG_generated = compute_CoG(function_code_string)

        response_debug= await runtime.send_message(TestExecCodeMessage(prompt, entry_point, function_code_string, test), AgentId("test_executor", "default"))

        # The function generated is already correct
        if response_debug.passed and response_debug.attempts == 0:
            passed=True
        else:
            passed = False

        # Canonical solution execution
        canonical_code = prompt +  canonical_solution
        CC_canonical = compute_CC(canonical_code)
        CoG_canonical = compute_CoG(canonical_code)

        new_data = {
            'task_id': [str(task_id)],
            'passed': [passed],
            'passed after debugging': [response_debug.passed],
            'tokens': [response.tokens],
            'generation time': [response.time],
            'debugging time': [response_debug.time],
            'attempts': [response_debug.attempts],
            'CC generation': [CC_generated],
            'CC canonical': [CC_canonical],
            'CoG generation': [CoG_generated],
            'CoG canonical': [CoG_canonical],
            'Original function': [function_code_string],
            'Debugged function': [response_debug.final_function]
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
        model_client = None

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


