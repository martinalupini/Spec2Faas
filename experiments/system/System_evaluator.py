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



async def main(config, models, server):
    work_dir = tempfile.mkdtemp()
    runtime = SingleThreadedAgentRuntime()

    attempt = config['experiment_number']
    llm = config['llm']

    ################################## REGISTERING AGENTS ###############################################
    await Assistant.register(runtime, "assistant", lambda: Assistant(llm=llm['assistant'], model_client=models['assistant']))
    await EntryPoint.register(runtime, "entry_point",
                              lambda: EntryPoint(llm=llm['entry_point'], model_client=models['entry_point']))
    await Coder.register(runtime, "coder", lambda: Coder(llm=llm['coder'], model_client=models['coder']))
    await TestDesigner.register(runtime, "test_designer",
                                lambda: TestDesigner(llm=llm['test_designer'], model_client=models['test_designer']))
    await Debugger.register(runtime, "debugger", lambda: Debugger(llm=llm['debugger'], model_client=models['debugger']))
    # creating the tools for the FaaS deployer
    tools: List[Tool] = [
        FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge and deploy the function on Serveledge.")]
    await FaasDeployer.register(runtime, "faas_deployer",
                                lambda: FaasDeployer(llm=llm['faas_deployer'], model_client=models['faas_deployer'], tool_schema=tools))
    # Registering the Test Executor
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    # This method sets the working environment variables, connects to Docker and starts the code executor. If no working directory was provided to the code executor, it creates a temporary directory and sets it as the code executor working directory.
    # https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.code_executors.docker.html#autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor
    await executor.start()
    await TestExecutor.register(runtime, "test_executor",
                                lambda: TestExecutor(llm=llm['test_executor'], model_client=models['test_executor'],
                                                     code_executor=executor))

    ############################### IMPORTING DATASET AND CREATING FILES ###################################
    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    # Creating files to store data
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
        'token_deployer', 'tests', 'coverage', 'debugging_attempts', 'number_messages_exchanged', 'canonical_solution', 'deployed_function'
    ]

    if os.path.exists(file_name):
        results_df = pd.read_parquet(file_name)
    else:
        results_df = pd.DataFrame(columns=columns)

    runtime.start()  # Start processing messages in the background.

    ################################# STARTING TESTS ########################################
    # Iterating through each row
    for row in df.itertuples(index=False):
        task_id = row.task_id

        # Function already generated in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        entry_point = row.entry_point
        prompt = row.prompt
        test = row.test
        canonical_solution = row.canonical_solution
        canonical_code = prompt + canonical_solution
        tokens = {'assistant': 0.0, 'entry_point': 0.0, 'coder': 0.0, 'test_designer': 0.0, 'test_executor': 0.0,
                        'debugger': 0.0,
                        'faas_deployer': 0.0}
        time = {'assistant': 0.0, 'entry_point': 0.0, 'coder': 0.0, 'test_designer': 0.0, 'test_executor': 0.0,
                      'debugger': 0.0,
                      'faas_deployer': 0.0, 'system': 0.0}

        correctly_executed = False
        original_function_correct = False
        final_function_correct = False
        test_correct = False
        CC_final = 0
        CoG_final = 0
        CC_original = 0
        CoG_original = 0
        coverage = 0
        json_created = True
        json_filename = entry_point + ".json"
        local_path = "system_results/"+ "experiment_" + str(attempt) + "/inputs/" + task_id.split('/')[-1] + "_" + json_filename
        remote_path = "serverledge/inputs_experiment_" + str(attempt) + "/" + json_filename

        # Creating directory if it does not exist
        directory = os.path.dirname(local_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        canonical_code = prompt + canonical_solution
        CC_canonical = compute_CC(canonical_code)
        CoG_canonical = compute_CoG(canonical_code)

        print_yellow(task_id)
        # Start the test
        response = await runtime.send_message(TestSystemMessage(tokens = tokens, time = time, prompt=prompt), AgentId("assistant", "default"))

        ######################## TESTING IF ORIGINAL FUNCTION IS CORRECT ######################################

        result, execution_time_generated = await execute_function(response.original_func, test, entry_point,
                                                                  executor, CancellationToken())
        if "AssertionError" in result.output:
            original_function_correct = False
        else:
            original_function_correct = True

        ######################## TESTING IF GENERATED TESTS ARE CORRECT ######################################
        function_name_code = extract_function_name(canonical_code)
        function_name_test = extract_function_name(response.signature)
        if function_name_code != function_name_test:
            new_canonical_code = canonical_code.replace(function_name_code, function_name_test)
        else:
            new_canonical_code = canonical_code
        result, execution_time_generated = await execute_tests(new_canonical_code, response.tests_str, executor,
                                                               CancellationToken())
        if "AssertionError" in result.output or "Error" in result.output:
            print_blue(str(result.output))
            test_correct = False
            coverage = 0
        else:
            test_correct = True
            match = re.search(r'(\d+)\s*%', result.output)
            if match:
                coverage = int(match.group(1))
            else:
                coverage = 0


        if response.generated:
            CC_original = compute_CC(response.original_func)
            CoG_original = compute_CoG(response.original_func)

            if response.deployed:

                ########################## TESTING IF CAN EXECUTE ON SERVERLEDGE ###########################
                # Need to save json file containing arguments. Cannot reuse the one from the deployer experiment because
                # the names of the arguments may be different.
                param_names = extract_param_names(response.signature, "", True)
                param_values = extract_param_values(test)

                try:
                    sftp = server.open_sftp()

                    try:
                        sftp.stat(remote_path)
                        continue
                    except FileNotFoundError:
                        create_json(param_names, param_values, local_path)
                        sftp.put(local_path, remote_path)
                        print("Json uploaded!")

                except Exception as e:
                    print(f"Error during file transfer: {e}")
                    json_created = False
                finally:
                    if sftp:
                        sftp.close()

                if json_created:
                    command = "serverledge/bin/serverledge-cli invoke -f " + response.result_deployment + " --params_file " + remote_path + " --ret_output"
                    stdin, stdout, stderr = server.exec_command(command)

                    output = stdout.read().decode('utf-8')
                    error = stderr.read().decode('utf-8')

                    if "\"Success\": true," in output:
                        correctly_executed = True

                    if output is not None:
                        print("--- Output ---")
                        print(output)

        ######################## TESTING IF FINAL FUNCTION IS CORRECT ######################################
            result, execution_time_generated = await execute_function(response.final_func, test, entry_point,
                                                                      executor, CancellationToken())
            if "AssertionError" in result.output:
                final_function_correct = False
            else:
                final_function_correct = True

            CC_final = compute_CC(response.final_func)
            CoG_final = compute_CoG(response.final_func)

        time = response.time
        tokens = response.tokens

        if response.attempts == 0:
            debugged = False
        else:
            debugged = True

        new_data = {
            'task_id': [str(task_id)],
            'generated': [response.generated],
            'deployed': [response.deployed],
            'correctly_executed': [correctly_executed],
            'debugged': [debugged],
            'original_function_correct': [original_function_correct],
            'final_function_correct': [final_function_correct],
            'test_correct': [test_correct],
            'prompt': [response.prompt],
            'signature': [response.signature],
            'original_function': [response.original_func],
            'final_function': [response.final_func],
            'CC_original': [CC_original],
            'CC_final': [CC_final],
            'CC_canonical': [CC_canonical],
            'CoG_final': [CoG_final],
            'CoG_original': [CoG_original],
            'CoG_canonical': [CoG_canonical],
            'time_assistant': [time['assistant']],
            'time_entry_point': [time['entry_point']],
            'time_coder': [time['coder']],
            'time_designer': [time['test_designer']],
            'time_executor': [time['test_executor']],
            'time_debugger': [time['debugger']],
            'time_deployer': [time['faas_deployer']],
            'time_system': [time['system']],
            'token_assistant': [tokens['assistant']],
            'token_entry_point': [tokens['entry_point']],
            'token_coder': [tokens['coder']],
            'token_designer': [tokens['test_designer']],
            'token_executor': [tokens['test_executor']],
            'token_debugger': [tokens['debugger']],
            'token_deployer': [tokens['faas_deployer']],
            'tests': [response.tests_str],
            'coverage': [coverage],
            'debugging_attempts': [response.attempts],
            'number_messages_exchanged': [response.messages],
            'canonical_solution': [canonical_code],
            'deployed_function': [response.deployed_function]
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
    for model in models.values():
        await model.close()





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

    # Connection to the server
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(hostname=SERVER_HOSTNAME, port=int(SERVER_PORT), username=SERVER_USERNAME,
                   password=SERVER_PASSWORD)

    try:
        asyncio.run(main(config, models, client))
    finally:
        logging.shutdown()