import pandas as pd
from app.Utils import *

llm = get_config_data("../../config_test.yaml")
coder = llm['coder']
if llm['coder_prompt'] == "Yes":
    prompt = True
else:
    prompt = False

if not prompt:
    coder = coder + "_no_prompt"

file_path = coder + ".parquet"
csv_file = "../results.csv"

try:
    df = pd.read_parquet(file_path)

    #print(df.head(10))

    #print(df.info())

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")


def write_csv(statistics=False):
    pass_at_1 = df['passed'].mean()
    num_generated = df['passed'].sum()
    avg_generation_time = df['generation_time'].mean()
    avg_tokens = df['tokens'].mean()
    avg_execution_time_generation = df['execution_time'].mean()
    avg_execution_time_canonical = df['execution_time_canonical'].mean()
    avg_cc_generation = df[df['passed'] == True]['CC_generation'].mean()
    avg_cc_canonical = df['CC_canonical'].mean()
    avg_cog_generation = df[df['passed'] == True]['CoG_generation'].mean()
    avg_cog_canonical = df['CoG_canonical'].mean()

    print(f"pass@1: {pass_at_1:.4f}")
    print(f"\nAverage Generation Time: {avg_generation_time:.4f}")
    print(f"\nAverage Tokens: {avg_tokens:.4f}")
    print("\n--- Average Execution Time ---")
    print(f"Generated function: {avg_execution_time_generation:.4f}")
    print(f"Canonical function: {avg_execution_time_canonical:.4f}")
    print("\n--- Average Cyclomatic Complexity (CC) ---")
    print(f"Generated function: {avg_cc_generation}")
    print(f"Canonical function: {avg_cc_canonical:.4f}")
    print("\n--- Average Cognitive Complexity (CoG) ---")
    print(f"Generated function: {avg_cog_generation:}")
    print(f"Canonical function: {avg_cog_canonical:.4f}")

    results_data = {
        'model': [coder],
        'pass@1': [pass_at_1],
        'number_func_generated': [num_generated],
        'avg_generation_time (s)': [avg_generation_time],
        'avg_tokens': [avg_tokens],
        'avg_execution_time (s)': [avg_execution_time_generation],
        'avg_cc_generation': [avg_cc_generation],
        'avg_cog_generation': [avg_cog_generation],
    }

    results_df = pd.DataFrame(results_data)

    if statistics:
        file_path_csv = '../results_deepseek.csv'
    else:
        file_path_csv = csv_file

    file_exists = os.path.exists(file_path_csv)

    try:
        # Usa mode='a' (append) per aggiungere dati alla fine del file.
        # L'intestazione viene scritta solo se il file non esiste (header=not file_exists).
        results_df.to_csv(file_path_csv, mode='a', header=not file_exists, index=False)

        if file_exists:
            print(f"\nAppended a new row to '{csv_file}'")
        else:
            print(f"\nCreated a new file and saved results to '{csv_file}'")

    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")



write_csv(True)