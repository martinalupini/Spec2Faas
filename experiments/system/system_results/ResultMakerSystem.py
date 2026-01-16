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
        'original_correct_final_correct': df[df['original_function_correct'] & df['final_function_correct']].shape[0],
        'original_func_correct' : df['original_function_correct'].sum(),
        'executed_and_final_correct': df[df['correctly_executed'] & df['final_function_correct']].shape[0],
        'executed_and_final_not_correct': df[df['correctly_executed'] & ~df['final_function_correct']].shape[0],
        'executed_and_original_correct': df[df['correctly_executed'] & df['original_function_correct']].shape[0],
        'original_correct_and_deployed': df[df['original_function_correct'] & df['deployed']].shape[0],
        'original_correct_and_not_deployed': df[df['original_function_correct'] & ~df['deployed']].shape[0],
        'final_func_correct': func_correct,
        'original_correct_final_not_correct': df[df['original_function_correct'] & ~df['final_function_correct']].shape[0],
        'original_correct_final_correct_debugged': df[df['original_function_correct'] & df['final_function_correct'] & df['debugged']].shape[0],
        'original_correct_final_correct_not_debugged': df[df['original_function_correct'] & df['final_function_correct'] & ~df['debugged']].shape[0],
        'original_not_correct_final_correct_debugged': df[~df['original_function_correct'] & df['final_function_correct'] & df['debugged']].shape[0],
        'original_not_correct_debugged': df[~df['original_function_correct'] & df['debugged']].shape[0],
        'original_not_correct_not_debugged': df[~df['original_function_correct'] & ~df['debugged']].shape[0],
        'original_not_correct_final_not_correct_debugged': df[~df['original_function_correct'] & ~df['final_function_correct'] & df['debugged']].shape[0],
        'original_not_correct_final_not_correct_not_debugged': df[~df['original_function_correct'] & ~df['final_function_correct'] & ~df['debugged']].shape[0],
        'original_correct_not_flagged_as_generated': df[df['original_function_correct'] & ~df['generated']].shape[0],
        'final_not_correct_deployed': df[df['deployed'] & ~df['final_function_correct']].shape[0],
        'final_not_correct_not_deployed': df[~df['deployed'] & ~df['final_function_correct']].shape[0],
        'broken_by_debug_correct_test': df[df['original_function_correct'] & ~df['final_function_correct'] & df['test_correct']].shape[0],
        'broken_by_debug_not_correct_test': df[df['original_function_correct'] & ~df['final_function_correct'] & ~df['test_correct']].shape[0],
        'final_correct_deployed': df[df['deployed'] & df['final_function_correct']].shape[0],
        'final_correct_not_deployed': df[~df['deployed'] & df['final_function_correct']].shape[0]

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


write_csv()


