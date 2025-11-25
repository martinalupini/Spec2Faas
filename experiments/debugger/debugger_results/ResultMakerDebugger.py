import pandas as pd
from app.Utils import *

llm = get_config_data("../../config_test.yaml")
debugger = llm['debugger']
coder = llm['coder']


def write_csv():
    file_path = coder + "_" + debugger + ".parquet"
    csv_file = "../results.csv"

    try:
        df = pd.read_parquet(file_path)

        # print(df.head(10))

        # print(df.info())

    except FileNotFoundError:
        print(f"Errore: File non trovato a questo percorso: {file_path}")
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")

    passed_after_generation = df['passed'].mean()
    number_passed_after_generation = df['passed'].sum()
    passed_after_debugging = df['passed_after_debugging'].mean()
    number_passed_after_debugging = df['passed_after_debugging'].sum()
    avg_tokens = df['total_tokens'].mean()
    avg_debug_tokens = df[df['passed'] == False]['debugging_tokens'].mean()
    avg_generation_time = df['generation_time'].mean()
    avg_debugging_time = df[df['passed'] == False]['debugging_time'].mean()
    avg_attempts_debugging = df[df['passed'] == False]['attempts'].mean()
    avg_CC_debugged = df[df['passed'] == False]['CC_debugged'].mean()
    avg_CoG_debugged = df[df['passed'] == False]['CoG_debugged'].mean()


    print(f"Passed after generation: {passed_after_generation:.4f}")
    print(f"\nNumber passed after generation: {number_passed_after_generation:.4f}")
    print(f"\nAverage Generation Time: {avg_generation_time:.4f}")
    print(f"\nAverage Debugging Time: {avg_debugging_time:.4f}")
    print(f"\nAverage Tokens: {avg_tokens:.4f}")
    print(f"\nAverage Debugging Tokens: {avg_debug_tokens:.4f}")
    print(f"\nPassed after debugging: {passed_after_debugging:.4f}")
    print(f"\nNumber passed after debugging: {number_passed_after_debugging:.4f}")
    print(f"\nAverage attempts debugging: {avg_attempts_debugging:.4f}")
    print(f"\nAverage CC_debugged: {avg_CC_debugged}")
    print(f"\nAverage CoG_debugged: {avg_CoG_debugged}")

    results_data = {
        'coder': [coder],
        'debugger': [debugger],
        'passed_after_generation': [passed_after_generation],
        'number_passed_after_generation': [number_passed_after_generation],
        'avg_generation_time (s)': [avg_generation_time],
        'average_debugging_time (s)': [avg_debugging_time],
        'average_total_time (s)': [avg_generation_time + avg_debugging_time],
        'avg_debugging_tokens': [avg_debug_tokens],
        'avg_total_tokens': [avg_tokens],
        'passed_after_debugging': [passed_after_debugging],
        'number_passed_after_debugging': [number_passed_after_debugging],
        'avg_attempts_debugging': [avg_attempts_debugging],
        'avg_CC_debugged': [avg_CC_debugged],
        'avg_CoG_debugged': [avg_CoG_debugged],
    }

    results_df = pd.DataFrame(results_data)

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
