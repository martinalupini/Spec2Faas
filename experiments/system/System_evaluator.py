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

async def main(llm, models, system_prompt):
    work_dir = tempfile.mkdtemp()
    runtime = SingleThreadedAgentRuntime()
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
    await model_client.close()





if __name__ == "__main__":

    load_env_variables()
    log_path = "../../output/designer/log"
    set_logging_config(log_path)
    llm = get_config_data("../config_test.yaml")
    coder = llm['coder']
    designer = llm['test_designer']
    debugger = llm['debugger']
    assistant = llm['assistant']
    entry_point = llm['entry_point']
    faas_deployer = llm['faas_deployer']
    test_executor = llm['test_executor']
    if llm['coder_prompt'] == "Yes":
        prompt = True
    else:
        prompt = False
    models = {}

    if coder == "gemini-2.5-pro" or coder == "gemini-2.0-flash":
        model_coder = OpenAIChatCompletionClient(
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
        model_coder = OllamaChatCompletionClient(
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

    models['coder'] = model_coder

    if assistant == "gemini-2.5-pro" or assistant == "gemini-2.0-flash":
        model_assistant = OpenAIChatCompletionClient(
            model=assistant,
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
        model_assistant = OllamaChatCompletionClient(
            model=assistant,
            host="160.80.97.151:11434",
            model_info={
                "family": ModelFamily.UNKNOWN,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            }
        )

    models['assistant'] = model_assistant

    if entry_point == "gemini-2.5-pro" or entry_point == "gemini-2.0-flash":
        model_entry = OpenAIChatCompletionClient(
            model=entry_point,
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
        model_entry = OllamaChatCompletionClient(
            model=entry_point,
            host="160.80.97.151:11434",
            model_info={
                "family": ModelFamily.UNKNOWN,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            }
        )

    models['entry_point'] = model_entry

    if debugger == "gemini-2.5-pro" or debugger == "gemini-2.0-flash":
        model_debugger = OpenAIChatCompletionClient(
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
        model_debugger = OllamaChatCompletionClient(
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

    models['debugger'] = model_debugger

    if test_executor == "gemini-2.5-pro" or test_executor == "gemini-2.0-flash":
        model_executor = OpenAIChatCompletionClient(
            model=test_executor,
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
        model_executor = OllamaChatCompletionClient(
            model=test_executor,
            host="160.80.97.151:11434",
            model_info={
                "family": ModelFamily.UNKNOWN,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            }
        )

    models['test_executor'] = model_executor

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

    models['test_designer'] = model_client

    if faas_deployer == "gemini-2.5-pro" or faas_deployer == "gemini-2.0-flash":
        model_deployer = OpenAIChatCompletionClient(
            model=faas_deployer,
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
        model_deployer = OllamaChatCompletionClient(
            model=faas_deployer,
            host="160.80.97.151:11434",
            model_info={
                "family": ModelFamily.UNKNOWN,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            }
        )

    models['faas_deployer'] = model_deployer

    try:
        asyncio.run(main(llm, models, prompt))
    finally:
        logging.shutdown()


