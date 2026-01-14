import asyncio
import tempfile
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from app.agents.coding_agents.utils.Utils import *
from app.agents.coding_agents.utils.Code_Extractors import *
from app.Utils import *
from experiments.MessageTypesTest import *
import pandas as pd
import time


async def check_tests(function: str, test:str, executor, ctx):
    # Installing dependencies in container
    dependencies = "```sh\npip install numpy```"
    code_block = extract_markdown_code_blocks(dependencies)

    code =  function + test
    invocation_code = CodeBlock(code=code, language='python')
    code_block.append(invocation_code)

    start_time = time.perf_counter()
    result = await executor.execute_code_blocks(code_block, ctx)
    end_time = time.perf_counter()

    return result, end_time - start_time


async def execute_function(function: str, test:str, executor, ctx):
    # Installing dependencies in container
    full_script = f"{function}\n\n{test}"


    dependencies = "```sh\npip install numpy\npip install coverage```"
    code_block = extract_markdown_code_blocks(dependencies)

    code = f"""\nimport os\nimport sys\ndef run():\n    try:\n        with open('solution.py', 'w') as f:\n           f.write({repr(full_script)})\n    except Exception as e:\n        return\n    exit_code = os.system("coverage run --source=. solution.py")\n    os.system("coverage report -m")\nrun()"""
    invocation_code = CodeBlock(code=code, language='python')
    code_block.append(invocation_code)


    result = await executor.execute_code_blocks(code_block, ctx)

    print(result.output)

    match = re.search(r'(\d+)\s*%', result.output)
    if match:
        coverage = int(match.group(1))
    else:
        # Cover the cases where the output is nor an AssertionError neither a coverage report
        coverage = 0

    print(coverage)

    return coverage



async def main(llm):
    work_dir = tempfile.mkdtemp()
    executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
    ctx = CancellationToken()
    await executor.start()

    # Importing the dataset
    humaneval = pd.read_parquet("hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet")
    file_name = "designer_results/" + llm + ".parquet"
    df = pd.read_parquet(file_name)

    for index, row in df.iterrows():
        if row['passed']:
            task_id = row['task_id']
            print_yellow(task_id)

            he_row = humaneval[humaneval['task_id'] == task_id]
            code = he_row['prompt'].values[0] + he_row['canonical_solution'].values[0]
            tests = row['tests']

            result, execution_time_generated = await check_tests(code, tests, executor,
                                                                      ctx)
            if "Error" in result.output:
                passed = False
                coverage_value = 0
            else:
                passed = True



                coverage_value = await execute_function(code, tests, executor, ctx)

            df.at[index, 'coverage'] = coverage_value
            df.at[index, 'passed'] = passed

            print(f"Task {task_id}: Coverage {coverage_value}% saved.")

    df.to_parquet(file_name, index=False)

    await executor.stop()



if __name__ == "__main__":

    load_env_variables()
    llm = get_config_data("../config_test.yaml")
    designer = llm['test_designer']
    try:
        asyncio.run(main(designer))
    finally:
        logging.shutdown()

