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

"""
It executes the function given the generated tests. It also computes the coverage
"""
async def execute_tests(function: str, test:str, executor, ctx):
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



async def main(config, llm_model):
    work_dir = tempfile.mkdtemp()
    runtime = SingleThreadedAgentRuntime()

    attempt = config['experiment_number']
    llm = config['llm']

    ################################## REGISTERING AGENTS ###############################################
    # creating the tools for the FaaS deployer
    tools: List[Tool] = [
        FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge and deploy the function on Serveledge.")]
    await FaasDeployer.register(runtime, "faas_deployer",
                                lambda: FaasDeployer(llm=llm['faas_deployer'], model_client=llm_model, tool_schema=tools))

    ############################### IMPORTING DATASET AND CREATING FILES ###################################
    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    # Creating files to store data
    old_file_name = "system_results/" + "experiment_" + str(attempt-1) + "/results.parquet"
    file_name = "system_results/"+ "experiment_" + str(attempt) + "/results.parquet"
    config_file_name = "system_results/"+ "experiment_" + str(attempt) + "/config.yaml"

    directory = os.path.dirname(config_file_name)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Storing current configuration
    save_yaml(config_file_name, config)

    columns = [
        'task_id', 'generated', 'deployed', 'correctly_executed', 'debugged', 'original_function_correct',
        'final_function_correct', 'test_correct' , 'prompt', 'signature', 'original_function', 'final_function',
        'CC_original', 'CC_final', 'CC_canonical', 'CoG_final', 'CoG_generated', 'CoG_canonical', 'time_assistant',
        'token_assistant', 'time_entry_point', 'token_entry_point', 'time_coder', 'token_coder', 'time_designer',
        'token_designer', 'time_executor', 'token_executor','time_debugger', 'token_debugger', 'time_deployer',
        'token_deployer', 'tests', 'coverage', 'debugging_attempts', 'number_messages_exchanged', 'canonical_solution', 'deployed_function', 'execution_output'
    ]

    old_results_df = pd.read_parquet(old_file_name)

    if os.path.exists(file_name):
        results_df = pd.read_parquet(file_name)
    else:
        results_df = pd.DataFrame(columns=columns)

    runtime.start()  # Start processing messages in the background.

    ################################# STARTING TESTS ########################################
    # Iterating through each row
    for row in df.itertuples(index=False):
        task_id = row.task_id
        entry_point = row.entry_point
        test = row.test

        deployed = False
        correctly_executed = False
        deployed_function = ""
        time_deployer = 0
        tokens_deployer = 0
        number_messages_exchanged = 0
        execution_output = ""

        # Function already generated in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        print_yellow(task_id)

        old_row = old_results_df.loc[old_results_df['task_id'] == task_id].iloc[0]

        if old_row['generated']:
            json_filename = entry_point + ".json"
            local_path = "system_results/"+ "experiment_" + str(attempt) + "/inputs/" + task_id.split('/')[-1] + "_" + json_filename


            # Creating directory if it does not exist
            directory = os.path.dirname(local_path)
            if directory:
                os.makedirs(directory, exist_ok=True)


            deploy_mess =  await runtime.send_message(TestDeployMessage(old_row['final_function']), AgentId("faas_deployer", "default"))
            tokens_deployer = deploy_mess.tokens
            time_deployer = deploy_mess.time

            if deploy_mess.result != "FAIL":
                deployed = True
                deployed_function = deploy_mess.deployed_function
            else:
                # In case the deployment is not successful
                deployed = False


            if deployed:

                ########################## TESTING IF CAN EXECUTE ON SERVERLEDGE ###########################
                # Need to save json file containing arguments. Cannot reuse the one from the deployer experiment because
                # the names of the arguments may be different.
                param_names = extract_param_names(old_row['signature'], "", True)
                param_values = extract_param_values(test)

                created_json = create_json(param_names, param_values, local_path)

                payload = {
                    "Params": created_json
                }

                headers = {"Content-Type": "application/json"}
                url = os.getenv("SERVERLEDGE_URL") + "/invoke/" + str(deploy_mess.result)
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        resp_text = await resp.text()
                        print(f"Resp status: {resp.status} Resp text: {resp_text}")
                        if resp.status == 200:
                            correctly_executed = True
                        execution_output = resp_text


        new_data = {
            'task_id': [str(task_id)],
            'generated': [old_row['generated']],
            'deployed': [deployed],
            'correctly_executed': [correctly_executed],
            'debugged': [old_row['debugged']],
            'original_function_correct': [old_row['original_function_correct']],
            'final_function_correct': [old_row['final_function_correct']],
            'test_correct': [old_row['test_correct']],
            'prompt': [old_row['prompt']],
            'signature': [old_row['signature']],
            'original_function': [old_row['original_function']],
            'final_function': [old_row['final_function']],
            'CC_original': [old_row['CC_original']],
            'CC_final': [old_row['CC_final']],
            'CC_canonical': [old_row['CC_canonical']],
            'CoG_final': [old_row['CoG_final']],
            'CoG_original': [old_row['CoG_original']],
            'CoG_canonical': [old_row['CoG_canonical']],
            'time_assistant': [old_row['time_assistant']],
            'time_entry_point': [old_row['time_entry_point']],
            'time_coder': [old_row['time_coder']],
            'time_designer': [old_row['time_designer']],
            'time_executor': [old_row['time_executor']],
            'time_debugger': [old_row['time_debugger']],
            'time_deployer': [time_deployer],
            'time_system': [old_row['time_system']],
            'token_assistant': [old_row['token_assistant']],
            'token_entry_point': [old_row['token_entry_point']],
            'token_coder': [old_row['token_coder']],
            'token_designer': [old_row['token_designer']],
            'token_executor': [old_row['token_executor']],
            'token_debugger': [old_row['token_debugger']],
            'token_deployer': [tokens_deployer],
            'tests': [old_row['tests']],
            'coverage': [old_row['coverage']],
            'debugging_attempts': [old_row['debugging_attempts']],
            'number_messages_exchanged': [number_messages_exchanged],
            'canonical_solution': [old_row['canonical_solution']],
            'deployed_function': [deployed_function],
            'execution_output': [execution_output]
        }

        new_row_df = pd.DataFrame(new_data)
        if results_df.empty:
            results_df = new_row_df
        else:
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        results_df['task_id'] = results_df['task_id'].astype('string')
        results_df.to_parquet(file_name, engine='pyarrow')


    await runtime.stop()  # Stop processing messages in the background.
    await llm_model.close()





if __name__ == "__main__":

    load_env_variables()
    log_path = "../../output/system/log"
    set_logging_config(log_path)
    config = get_config_data_full("../config_test.yaml")
    llm = config['llm']
    SERVER_HOSTNAME = os.environ.get("SERVERLEDGE_IP")
    SERVER_PORT = os.environ.get("SERVERLEDGE_PORT")
    SERVER_USERNAME = os.environ.get("SERVERLEDGE_USERNAME")
    SERVER_PASSWORD = os.environ.get("SERVERLEDGE_PASS")

    llm_deployer = llm['faas_deployer']


    if llm_deployer == "gemini-2.5-pro" or llm_deployer == "gemini-2.0-flash":
        llm_model = OpenAIChatCompletionClient(
            model=llm_deployer,
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
            model=llm_deployer,
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
        asyncio.run(main(config, llm_model))
    finally:
        logging.shutdown()