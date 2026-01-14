import numpy as np
import pandas as pd
import os
from app.Utils import *
import matplotlib.pyplot as plt
import math

llm = get_config_data("../../config_test.yaml")
designer = llm['test_designer']

file_path = designer + ".parquet"
csv_file = "../results.csv"

try:
    df = pd.read_parquet(file_path)

    #print(df.head(10))

    #print(df.info())

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")


def write_csv():
    passed = df['passed'].mean()
    num_passed = df['passed'].sum()
    avg_generation_time = df['generation_time'].mean()
    avg_tokens = df['tokens'].mean()
    avg_execution_time_generation = df['execution_time'].mean()
    avg_coverage = df[df['passed'] == True]['coverage'].mean()


    print(f"Passed: {passed}")
    print(f"Num passed: {num_passed}")
    print(f"\nAverage Generation Time: {avg_generation_time:.4f}")
    print(f"\nAverage Tokens: {avg_tokens:.4f}")
    print(f"Average Execution Time: {avg_execution_time_generation:.4f}")
    print(f"\nAverage Coverage: {avg_coverage}")

    results_data = {
        'model': [designer],
        'passed': [passed],
        'num_passed': [num_passed],
        'avg_test_generation_time (s)': [avg_generation_time],
        'avg_tokens': [avg_tokens],
        'avg_test_execution_time (s)': [avg_execution_time_generation],
        'avg_coverage (%)': [avg_coverage]
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
