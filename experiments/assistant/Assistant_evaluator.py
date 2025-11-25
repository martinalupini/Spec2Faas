import logging
import asyncio
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from app.agents.Assistant import *
from app.agents.coding_agents.utils.Utils import *
from app.Utils import *
import pandas as pd


async def main(llm, client):

    runtime = SingleThreadedAgentRuntime()
    await Assistant.register(runtime, "assistant", lambda: Assistant(llm=llm, model_client=client))

    #Importing the dataset
    df = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")

    # Creating file to store data
    file_name = "assistant_results/"+ llm+".parquet"

    columns = [
        'task_id', 'actual_type', 'type', 'time', 'tokens',
        'response'
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
        error = ""

        print_yellow(task_id)

        message_to_send = prompt
        response= await runtime.send_message(TestMessage(message_to_send), AgentId("assistant", "default"))


        new_data = {
            'task_id': [str(task_id)],
            'actual_type': ["translation"],
            'type': [response.type],
            'time': [response.time],
            'tokens': [response.tokens],
            'response': [response.content]
        }

        new_row_df = pd.DataFrame(new_data)
        if results_df.empty:
            results_df = new_row_df
        else:
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        message_to_send = prompt + canonical_solution
        response = await runtime.send_message(TestMessage(message_to_send), AgentId("assistant", "default"))

        new_data = {
            'task_id': [str(task_id)],
            'actual_type': ["deployment"],
            'type': [response.type],
            'time': [response.time],
            'tokens': [response.tokens],
            'response': [response.content]
        }

        new_row_df = pd.DataFrame(new_data)
        if results_df.empty:
            results_df = new_row_df
        else:
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        results_df['task_id'] = results_df['task_id'].astype('string')
        results_df.to_parquet(file_name, engine='pyarrow')

    await runtime.stop()  # Stop processing messages in the background.
    await model_client.close()





if __name__ == "__main__":

    load_env_variables()
    log_path = "../../output/assistant/log"
    set_logging_config(log_path)
    llm = get_config_data("../config_test.yaml")
    assistant = llm['assistant']


    if assistant == "gemini-2.5-pro" or assistant == "gemini-2.0-flash":
        model_client = OpenAIChatCompletionClient(
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
        model_client = OllamaChatCompletionClient(
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

    try:
        asyncio.run(main(assistant, model_client))
    finally:
        logging.shutdown()


