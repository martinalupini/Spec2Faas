import logging
import asyncio
import sys
import tempfile
import time
import codecs
import paramiko
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.FaaSDeployer import *
from autogen_core.tools import FunctionTool, Tool
from app.agents.coding_agents.Coder import *
from app.agents.coding_agents.utils.Utils import *
from app.agents.coding_agents.utils.Code_Extractors import *
from app.Utils import *
import pandas as pd
from Utils import *



async def main(llm, client):

    runtime = SingleThreadedAgentRuntime()
    tools: List[Tool] = [FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge and deploy the function on Serveledge.")]
    await FaasDeployer.register(runtime, "faas_deployer", lambda: FaasDeployer(llm = llm, model_client=client, tool_schema=tools))

    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")


    file_name = "deployer_results/"+ llm+".parquet"
    columns = [
        'task_id', 'tokens',
        'deployment time', 'deployed', 'correctly executed', 'invocation_attempts',
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
        test = row.test
        prompt = row.prompt
        canonical_solution = row.canonical_solution
        json_filename = entry_point + ".json"
        local_path = "inputs/"+ task_id.split('/')[-1] + "_" + json_filename

        # Function already seen in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        # First of all, saving (locally and remotely) the json files containing the inputs (if not yet saved)
        if not os.path.exists(local_path):
            # I need to create a json file with parameter_name : parameter_value
            # The parameter names are taken from the prompt.
            param_names = extract_param_names(prompt, entry_point)
            # The parameter values are taken from the tests
            param_values = extract_param_values(test)

            create_json(param_names, param_values, local_path)

        with open(local_path, 'r') as file:
            created_json = json.load(file)

        print_yellow(task_id)

        # Extracting canonical solution to deploy
        canonical_function_code = prompt + canonical_solution
        response= await runtime.send_message(TestDeployMessage(canonical_function_code), AgentId("faas_deployer", "default"))
        correctly_executed = False

        # If deployed, test the correctness by invoking the function on Serverledge
        if response.result != 'FAIL':
            deployed = True

            payload = {
                "Params": created_json
            }

            headers = {"Content-Type": "application/json"}
            url = os.getenv("SERVERLEDGE_URL") + "/invoke/" + str(response.result)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    print(f"Resp status: {resp.status} Resp text: {await resp.text()}")
                    if resp.status == 200:
                        correctly_executed = True
        else:
            print("Failed to deploy.")
            deployed = False


        new_data = {
            'task_id': [str(task_id)],
            'tokens': [response.tokens],
            'deployment time': [response.time],
            'deployed': [deployed],
            'correctly executed': [correctly_executed],
            'invocation attempts': [response.invocation_attempts]
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
    SERVER_HOSTNAME = os.environ.get("SERVERLEDGE_IP")
    SERVER_PORT = os.environ.get("SERVERLEDGE_PORT")
    SERVER_USERNAME = os.environ.get("SERVERLEDGE_USERNAME")
    SERVER_PASSWORD = os.environ.get("SERVERLEDGE_PASS")
    log_path = "../../output/deployer/log"
    set_logging_config(log_path)
    llm = get_config_data("../config_test.yaml")
    deployer = llm['faas_deployer']

    if deployer == "gemini-2.5-pro" or deployer == "gemini-2.0-flash":
        model_client = OpenAIChatCompletionClient(
            model=deployer,
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
            model=deployer,
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
        asyncio.run(main(deployer, model_client))
    finally:
        logging.shutdown()