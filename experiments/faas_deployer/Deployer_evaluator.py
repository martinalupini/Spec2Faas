import logging
import asyncio
import sys
import tempfile
import time
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



async def main(llm, client, server):

    runtime = SingleThreadedAgentRuntime()
    tools: List[Tool] = [FunctionTool(create_json_serverledge, description="Create the json payload for a request for Serverledge.")]
    await FaasDeployer.register(runtime, "faas_deployer", lambda: FaasDeployer(llm = llm, model_client=client, tool_schema=tools))

    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")


    file_name = "deployer_results/"+ llm+".parquet"
    columns = [
        'task_id', 'tokens',
        'deployment time', 'deployed', 'correctly executed'
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
        remote_path = "serverledge/inputs" + "/" + json_filename

        # Function already seen in a previous experiment
        if task_id in results_df['task_id'].values:
            continue

        # First of all, saving the json files containing the inputs (if not yet saved)
        if not os.path.exists(local_path):
            param_names = extract_param_names(prompt, entry_point)
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
                raise e
            finally:
                if sftp:
                    sftp.close()

        print_yellow(task_id)

        # Extracting canonical solution
        canonical_function_code = prompt + canonical_solution
        response= await runtime.send_message(TestDeployMessage(canonical_function_code), AgentId("faas_deployer", "default"))
        correctly_executed = False
        if response.result != 'FAIL':
            deployed = True

            command = "serverledge/bin/serverledge-cli invoke -f " + response.result +" --params_file " + remote_path + " --ret_output"
            stdin, stdout, stderr = server.exec_command(command)

            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')

            if "\"Success\": true," in output:
                correctly_executed = True

            if output is not None:
                print("--- Output ---")
                print(output)
            
        else:
            deployed = False


        new_data = {
            'task_id': [str(task_id)],
            'tokens': [response.tokens],
            'deployment time': [response.time],
            'deployed': [deployed],
            'correctly executed': [correctly_executed]
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
            host=" 160.80.97.151:11434",
            model_info={
                "family": ModelFamily.UNKNOWN,
                "function_calling": True,
                "json_output": True,
                "vision": False,
                "structured_output": True,
            }
        )

    # Connection to the server
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(hostname=SERVER_HOSTNAME, port=int(SERVER_PORT), username=SERVER_USERNAME, password=SERVER_PASSWORD)

    try:
        asyncio.run(main(deployer, model_client, client))
    finally:
        if client:
            client.close()
        logging.shutdown()