from app.Utils import *
from experiments.faas_deployer.Utils import *


config = get_config_data_full("../../config_test.yaml")
experiment = config["experiment_number"]

directory = "experiment_" + str(experiment) + "/"
file_path = directory + "results.parquet"
csv_file = "../results.csv"


try:
    df = pd.read_parquet(file_path)

    #print(df.info())

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")



columns = [
        'task_id', 'generated', 'deployed', 'correctly_executed', 'debugged', 'original_function_correct',
        'final_function_correct', 'test_correct' , 'prompt', 'signature', 'original_function', 'final_function',
        'CC_original', 'CC_final', 'CC_canonical', 'CoG_final', 'CoG_generated', 'CoG_canonical', 'time_assistant',
        'token_assistant', 'time_entry_point', 'token_entry_point', 'time_coder', 'token_coder', 'time_designer',
        'token_designer', 'time_executor', 'token_executor','time_debugger', 'token_debugger', 'time_deployer',
        'token_deployer', 'tests', 'coverage', 'debugging_attempts', 'number_messages_exchanged', 'canonical_solution', 'deployed_function'
    ]


def write_csv():

    num_generated = df['generated'].sum()
    num_deployed = df['deployed'].sum()
    num_correctly_executed = df['correctly_executed'].sum()
    num_debugged = df['debugged'].sum()
    func_correct = df['final_function_correct'].sum()
    avg_tokens = df['total_tokens'].mean()
    avg_time = df['total_time'].mean()

    num_not_generated = 164 - num_generated
    num_generated_and_debugged = df[df['generated'] & df['debugged']].shape[0]
    num_generated_and_not_debugged = df[df['generated'] & ~df['debugged']].shape[0]
    num_debugged_not_generated = df[~df['generated'] & df['debugged']].shape[0]


    num_generated_debugged_deployed = df[df['generated'] & df['debugged'] & df['deployed']].shape[0]
    num_generated_not_debugged_deployed = df[df['generated'] & ~df['debugged'] & df['deployed']].shape[0]
    num_generated_debugged_not_deployed = df[df['generated'] & df['debugged'] & ~df['deployed']].shape[0]
    num_generated_not_debugged_not_deployed = df[df['generated'] & ~df['debugged'] & ~df['deployed']].shape[0]

    print(f"\n#Functions Generated: {num_generated}")
    print(f"\n#Functions Debugged: {num_debugged}")
    print(f"\n#Functions Deployed: {num_deployed}")
    print(f"\n#Functions Correctly Executed {num_correctly_executed}")
    print(f"\nAverage Total Tokens: {avg_tokens}")
    print(f"\nAverage Total Time: {avg_time}")


    results_data = {
        'experiment_number': experiment,
        'num_generated': num_generated,
        'num_deployed': num_deployed,
        'num_correctly_executed': num_correctly_executed,
        'num_debugged': num_debugged,
        'num_not_generated': num_not_generated,
        'num_generated_and_debugged': num_generated_and_debugged,
        'num_generated_and_not_debugged': num_generated_and_not_debugged,
        'num_generated_debugged_deployed': num_generated_debugged_deployed,
        'num_generated_not_debugged_deployed': num_generated_not_debugged_deployed,
        'num_generated_not_debugged_not_deployed': num_generated_not_debugged_not_deployed,
        'num_generated_debugged_not_deployed': num_generated_debugged_not_deployed,
        'num_debugged_not_generated': num_debugged_not_generated,
        'avg_total_tokens': avg_tokens,
        'avg_total_time': avg_time,
        'avg_tokens_assistant': df['token_assistant'].mean(),
        'avg_tokens_entry_point': df['token_entry_point'].mean(),
        'avg_tokens_coder': df['token_coder'].mean(),
        'avg_tokens_designer': df['token_designer'].mean(),
        'avg_tokens_executor': df['token_executor'].mean(),
        'avg_tokens_debugger': df['token_debugger'].mean(),
        'avg_tokens_deployer': df['token_deployer'].mean(),
        'avg_time_assistant': df['time_assistant'].mean(),
        'avg_time_entry_point': df['time_entry_point'].mean(),
        'avg_time_coder': df['time_coder'].mean(),
        'avg_time_designer': df['time_designer'].mean(),
        'avg_time_deployer': df['time_deployer'].mean(),
        'avg_time_executor': df['time_executor'].mean(),
        'avg_time_debugger': df['time_debugger'].mean(),
    }

    # Non so perchè in questo file serviva index altrimenti ho errore
    results_df = pd.DataFrame(results_data, index=[0])

    file_exists = os.path.exists(csv_file)

    try:
        # Usa mode='a' (append) per aggiungere dati alla fine del file.
        # L'intestazione viene scritta solo se il file non esiste (header=not file_exists).
        results_df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

        if file_exists:
            print(f"\nAppended a new row to '{csv_file}'")
        else:
            print(f"\nCreated a new file and saved results to '{csv_file}'")

    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")


def write_csv_sankey():
    file_path = "../results_sankey.csv"
    results_data = {
        'experiment': experiment,
        'deployed': df['deployed'].sum(),
        'correctly_executed': df['correctly_executed'].sum(),
        'deployed_not_executed': df[~df['correctly_executed'] & df['deployed']].shape[0],
        'executed_final_correct': df[df['correctly_executed'] & df['final_function_correct']].shape[0],
        'executed_final_not_correct': df[df['correctly_executed'] & ~df['final_function_correct']].shape[0],
        'executed_final_correct_debugged': df[df['correctly_executed'] & df['final_function_correct'] & df['debugged']].shape[0],
        'executed_final_correct_not_debugged': df[df['correctly_executed'] & df['final_function_correct'] & ~df['debugged']].shape[0],
        'executed_final_correct_debugged_original_correct': df[df['correctly_executed'] & df['final_function_correct'] & df['debugged'] & df['original_function_correct']].shape[0],
        'executed_final_correct_debugged_original_not_correct': df[df['correctly_executed'] & df['final_function_correct'] & df['debugged'] & ~df['original_function_correct']].shape[0],
        'executed_final_correct_not_debugged_original_correct': df[df['correctly_executed'] & df['final_function_correct'] & ~df['debugged'] & df['original_function_correct']].shape[0],
        'executed_final_correct_not_debugged_original_not_correct': df[df['correctly_executed'] & df['final_function_correct'] & ~df['debugged'] & ~df['original_function_correct']].shape[0],
        'executed_final_not_correct_debugged': df[df['correctly_executed'] & ~df['final_function_correct'] & df['debugged']].shape[0],
        'executed_final_not_correct_not_debugged': df[df['correctly_executed'] & ~df['final_function_correct'] & ~df['debugged']].shape[0],
        'executed_final_not_correct_debugged_original_correct': df[df['correctly_executed'] & ~df['final_function_correct'] & df['debugged'] & df['original_function_correct']].shape[0],
        'executed_final_not_correct_debugged_original_not_correct': df[df['correctly_executed'] & ~df['final_function_correct'] & df['debugged'] & ~df['original_function_correct']].shape[0],
        'executed_final_not_correct_not_debugged_original_correct': df[df['correctly_executed'] & ~df['final_function_correct'] & ~df['debugged'] & df['original_function_correct']].shape[0],
        'executed_final_not_correct_not_debugged_original_not_correct': df[df['correctly_executed'] & ~df['final_function_correct'] & ~df['debugged'] & ~df['original_function_correct']].shape[0],
        'not_executed_final_correct': df[df['deployed'] & ~df['correctly_executed'] & df['final_function_correct']].shape[0],
        'not_executed_final_not_correct': df[df['deployed'] & ~df['correctly_executed'] & ~df['final_function_correct']].shape[0],
        'not_executed_final_correct_debugged': df[df['deployed'] & ~df['correctly_executed'] & df['final_function_correct'] & df['debugged']].shape[0],
        'not_executed_final_correct_not_debugged': df[df['deployed'] & ~df['correctly_executed'] & df['final_function_correct'] & ~df['debugged']].shape[0],
        'not_executed_final_correct_debugged_original_correct': df[df['deployed'] & ~df['correctly_executed'] & df['final_function_correct'] & df['debugged'] & df['original_function_correct']].shape[0],
        'not_executed_final_correct_debugged_original_not_correct': df[df['deployed'] & ~df['correctly_executed'] & df['final_function_correct'] & df['debugged'] & ~df['original_function_correct']].shape[0],
        'not_executed_final_correct_not_debugged_original_correct': df[df['deployed'] & ~df['correctly_executed'] & df['final_function_correct'] & ~df['debugged'] & df['original_function_correct']].shape[0],
        'not_executed_final_correct_not_debugged_original_not_correct': df[df['deployed'] & ~df['correctly_executed'] & df['final_function_correct'] & ~df['debugged'] & ~df['original_function_correct']].shape[0],
        'not_executed_final_not_correct_debugged':df[df['deployed'] & ~df['correctly_executed'] & ~df['final_function_correct'] & df['debugged']].shape[0],
        'not_executed_final_not_correct_not_debugged':df[df['deployed'] & ~df['correctly_executed'] & ~df['final_function_correct'] & ~df['debugged']].shape[0],
        'not_executed_final_not_correct_debugged_original_correct': df[df['deployed'] & ~df['correctly_executed'] & ~df['final_function_correct'] & df['debugged'] & df['original_function_correct']].shape[0],
        'not_executed_final_not_correct_debugged_original_not_correct': df[df['deployed'] & ~df['correctly_executed'] & ~df['final_function_correct'] & df['debugged'] & ~df['original_function_correct']].shape[0],
        'not_executed_final_not_correct_not_debugged_original_correct': df[df['deployed'] & ~df['correctly_executed'] & ~df['final_function_correct'] & ~df['debugged'] & df['original_function_correct']].shape[0],
        'not_executed_final_not_correct_not_debugged_original_not_correct': df[df['deployed'] & ~df['correctly_executed'] & ~df['final_function_correct'] & ~df['debugged'] & ~df['original_function_correct']].shape[0],

    }

    # Non so perchè in questo file serviva index altrimenti ho errore
    results_df = pd.DataFrame(results_data, index=[0])

    file_exists = os.path.exists(file_path)

    try:
        # Usa mode='a' (append) per aggiungere dati alla fine del file.
        # L'intestazione viene scritta solo se il file non esiste (header=not file_exists).
        results_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

        if file_exists:
            print(f"\nAppended a new row to '{file_path}'")
        else:
            print(f"\nCreated a new file and saved results to '{file_path}'")

    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")



def write_csv_messages():

    file_path = "../results_messages.csv"


    num_messages = df['number_messages_exchanged'].mean()
    df_generated = df[df['generated']]
    df_generated_debugged = df[df['generated'] & df['debugged']]
    df_not_generated = df[~df['generated']]
    df_generated_not_debugged = df[df['generated'] & ~df['debugged']]

    results_data = {
        'experiment_number': experiment,
        'num_messages': num_messages,
        'num_messages_generated': df_generated['number_messages_exchanged'].mean(),
        'num_messages_not_generated': df_not_generated['number_messages_exchanged'].mean(),
        'num_messages_generated_debugged': df_generated_debugged['number_messages_exchanged'].mean(),
        'num_messages_generated_not_debugged': df_generated_not_debugged['number_messages_exchanged'].mean(),

    }

    results_df = pd.DataFrame(results_data, index=[0])

    file_exists = os.path.exists(file_path)

    try:
        results_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

        if file_exists:
            print(f"\nAppended a new row to '{file_path}'")
        else:
            print(f"\nCreated a new file and saved results to '{file_path}'")

    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")



#write_csv()
#write_csv_sankey()

write_csv_messages()


