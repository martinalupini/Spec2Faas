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

async def main(config, models, system_prompt, server):
    work_dir = tempfile.mkdtemp()
    runtime = SingleThreadedAgentRuntime()

    attempt = config['experiment_number']
    llm = config['llm']

    await Assistant.register(runtime, "assistant", lambda: Assistant(llm=llm['assistant'], model_client=models['assistant']))
    await EntryPoint.register(runtime, "entry_point",
                              lambda: EntryPoint(llm=llm['entry_point'], model_client=models['entry_point']))
    await Coder.register(runtime, "coder", lambda: Coder(llm=llm['coder'], model_client=models['coder']))
    await TestDesigner.register(runtime, "test_designer",
                                lambda: TestDesigner(llm=llm['test_designer'], model_client=models['test_designer']))
    await Debugger.register(runtime, "debugger", lambda: Debugger(llm=llm['debugger'], model_client=models['debugger']))
    # creating the tools for the FaaS deployer
    tools: List[Tool] = [
        FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge.")]
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

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    # Creating files to store data
    file_name = "system_results/"+ "experiment_" + str(attempt) + "/results.parquet"
    config_file_name = "system_results/"+ "experiment_" + str(attempt) + "/config.json"

    directory = os.path.dirname(config_file_name)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Storing current configuration
    save_yaml(config_file_name, config)

    columns = [
        'task_id', 'generated', 'deployed', 'correctly_executed', 'original_function_correct',
        'final_function_correct', 'test_correct' , 'prompt', 'signature', 'original_function', 'final_function',
        'CC_original', 'CC_final', 'CC_canonical', 'CoG_final', 'CoG_generated', 'CoG_canonical', 'time_assistant',
        'token_assistant', 'time_entry_point', 'token_entry_point', 'time_coder', 'token_coder', 'time_designer',
        'token_designer', 'time_executor', 'token_executor','time_debugger', 'token_debugger', 'time_deployer',
        'token_deployer', 'tests', 'coverage', 'debugging_attempts', 'number_messages_exchanged'
    ]

    if os.path.exists(file_name):
        results_df = pd.read_parquet(file_name)
    else:
        results_df = pd.DataFrame(columns=columns)

    runtime.start()  # Start processing messages in the background.

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
        messages = 1
        tokens = {'assistant': 0.0, 'entry_point': 0.0, 'coder': 0.0, 'test_designer': 0.0, 'test_executor': 0.0,
                        'debugger': 0.0,
                        'faas_deployer': 0.0}
        time = {'assistant': 0.0, 'entry_point': 0.0, 'coder': 0.0, 'test_designer': 0.0, 'test_executor': 0.0,
                      'debugger': 0.0,
                      'faas_deployer': 0.0}
        generated = False
        deployed = False
        correctly_executed = False
        original_function_correct = False
        final_function_correct = False
        test_correct = False
        original_function = ""
        final_function = ""
        tests = ""
        CC_final = 0
        CoG_final = 0
        CC_original = 0
        CoG_original = 0
        coverage = 0
        debugging_attempts = 0
        generated_prompt = ""
        signature = ""
        json_filename = entry_point + ".json"
        local_path = "../faas_deployer/inputs/" + task_id.split('/')[-1] + "_" + json_filename
        remote_path = "serverledge/inputs" + "/" + json_filename

        canonical_code = prompt + canonical_solution
        CC_canonical = compute_CC(canonical_code)
        CoG_canonical = compute_CoG(canonical_code)

        print_yellow(task_id)
        response_assistant = await runtime.send_message(TestMessage(prompt), AgentId("assistant", "default"))
        time['assistant'] = response_assistant.time
        tokens['assistant'] = response_assistant.tokens
        if response_assistant.content != "FAIL":
            generated_prompt = response_assistant.content
            messages += 1
            response_entry_point = await runtime.send_message(TestMessage(generated_prompt), AgentId("entry_point", "default"))
            time['entry_point'] = response_entry_point.time
            tokens['entry_point'] = response_entry_point.tokens
            signature = response_entry_point.content
            messages += 2
            response_coder = await runtime.send_message(TestCodeMessage(generated_prompt, signature, system_prompt), AgentId("coder", "default"))
            function_code = extract_markdown_code_blocks(response_coder.content)
            original_function = function_code[0].code
            time['coder'] = response_coder.time
            tokens['coder'] = response_coder.tokens
            messages += 1

            print(original_function)

            response_designer = await runtime.send_message(TestCodeMessage(generated_prompt, signature, system_prompt),
                                                  AgentId("test_designer", "default"))
            messages += 1
            time['designer'] = response_designer.time
            tokens['designer'] = response_designer.tokens
            tests = response_designer.content

            print(tests)
            messages += 1
            response_debug = await runtime.send_message(
                TestExecCodeMessage(prompt, signature, original_function, tests, system=True),
                AgentId("test_executor", "default"))
            messages += 1
            generated = response_debug.passed

            function_code = extract_markdown_code_blocks(response_coder.content)
            if function_code:
                final_function = function_code[0].code
            else:
                final_function = response_debug.final_function
            time['debugger'] = response_debug.time_debugger
            tokens['debugger'] = response_debug.tokens_debugger
            time['test_executor'] = response_debug.time
            tokens['test_executor'] = response_debug.tokens
            debugging_attempts = response_debug.attempts
            messages += debugging_attempts

            CC_original = compute_CC(original_function)
            CoG_original = compute_CoG(original_function)

            # If the function is correctly generated then I can compute the CC and CoG and see if it's correct
            if generated:

                response_deployer = await runtime.send_message(TestDeployMessage(final_function),
                                                               AgentId("faas_deployer", "default"))
                messages += 1
                tokens['faas_deployer'] = response_deployer.tokens
                time['faas_deployer'] = response_deployer.time


                if response_deployer.result != 'FAIL':
                    deployed = True

                    command = "serverledge/bin/serverledge-cli invoke -f " + response_deployer.result + " --params_file " + remote_path + " --ret_output"
                    stdin, stdout, stderr = server.exec_command(command)

                    output = stdout.read().decode('utf-8')
                    error = stderr.read().decode('utf-8')

                    if "\"Success\": true," in output:
                        correctly_executed = True

                    if output is not None:
                        print("--- Output ---")
                        print(output)

                result, execution_time_generated = await execute_function(final_function, test, entry_point,
                                                                          executor, CancellationToken())
                if "AssertionError" in result.output:
                    print_blue(f"\n{'-' * 130}\nExecutor:\n{result.output}\n{'-' * 130}")
                    final_function_correct = False
                else:
                    final_function_correct = True

                CC_final = compute_CC(final_function)
                CoG_final = compute_CoG(final_function)


            # Checking if the original function is correct
            result, execution_time_generated = await execute_function(original_function, test, entry_point,
                                                                      executor, CancellationToken())
            if "AssertionError" in result.output:
                print_blue(f"\n{'-' * 130}\nExecutor:\n{result.output}\n{'-' * 130}")
                original_function_correct = False
            else:
                original_function_correct = True


            # Checking if the generated tests are correct
            test_code = extract_markdown_code_blocks(response_designer.content)
            if test_code:
                tests = test_code[0].code
                result, execution_time_generated = await execute_tests(canonical_code, tests, executor,
                                                                          response_designer.ctx)
                if "AssertionError" in result.output:
                    test_correct = False
                    coverage = 0
                else:
                    test_correct = True
                    match = re.search(r'(\d+)\s*%', result.output)
                    if match:
                        coverage = int(match.group(1))
                    else:
                        coverage = 0
            else:
                test_correct = False

        new_data = {
            'task_id': [str(task_id)],
            'generated': [generated],
            'deployed': [deployed],
            'correctly_executed': [correctly_executed],
            'original_function_correct': [original_function_correct],
            'final_function_correct': [final_function_correct],
            'test_correct': [test_correct],
            'prompt': [generated_prompt],
            'signature': [signature],
            'original_function': [original_function],
            'final_function': [final_function],
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
            'token_assistant': [tokens['assistant']],
            'token_entry_point': [tokens['entry_point']],
            'token_coder': [tokens['coder']],
            'token_designer': [tokens['test_designer']],
            'token_executor': [tokens['test_executor']],
            'token_debugger': [tokens['debugger']],
            'token_deployer': [tokens['faas_deployer']],
            'tests': [tests],
            'coverage': [coverage],
            'debugging_attempts': [debugging_attempts],
            'number_messages_exchanged': [messages]
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
    for model in models:
        model.close()





if __name__ == "__main__":

    load_env_variables()
    log_path = "../../output/designer/log"
    set_logging_config(log_path)
    config = get_config_data_full("../config_test.yaml")
    llm = config['llm']
    SERVER_HOSTNAME = os.environ.get("SERVERLEDGE_IP")
    SERVER_PORT = os.environ.get("SERVERLEDGE_PORT")
    SERVER_USERNAME = os.environ.get("SERVERLEDGE_USERNAME")
    SERVER_PASSWORD = os.environ.get("SERVERLEDGE_PASS")

    if llm['coder_prompt'] == "Yes":
        prompt = True
    else:
        prompt = False
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
        asyncio.run(main(config, models, prompt, client))
    finally:
        logging.shutdown()