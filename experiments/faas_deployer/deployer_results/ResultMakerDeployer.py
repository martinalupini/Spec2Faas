import numpy as np
import pandas as pd
import os
from app.Utils import *
import matplotlib.pyplot as plt
import math

llm = get_config_data("../../config_test.yaml")
deployer = llm['faas_deployer']

file_path = deployer + ".parquet"
csv_file = "../results.csv"

try:
    df = pd.read_parquet(file_path)

    #print(df.head(10))

    #print(df.info())

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")

columns = [
        'task_id', 'tokens',
        'deployment time', 'deployed', 'correctly executed'
    ]

def write_csv():
    avg_tokens = df['tokens'].mean()
    avg_deployment_time = df['deployment time'].mean()
    avg_deployed = df['deployed'].mean()
    avg_executed = df['correctly executed'].mean()
    sum_deployed = df['deployed'].sum()
    sum_executed = df['correctly executed'].sum()
    avg_invocation_attempts = df['invocation attempts'].mean()


    print(f"\nAverage Tokens: {avg_tokens:.4f}")
    print(f"\nAverage Deployment Time: {avg_deployment_time:.4f}")
    print(f"\nFunctions Deployed: {avg_deployed:.4f}")
    print(f"\n#Functions Correctly Deployed: {sum_deployed}")
    print(f"\nFunctions Correctly Executed: {avg_executed:.4f}")
    print(f"\n#Functions Correctly Executed {sum_executed}")


    results_data = {
        'model': [deployer],
        'avg_deployment_time (s)': [avg_deployment_time],
        'avg_tokens': [avg_tokens],
        'functions_correctly_deployed': [avg_deployed],
        'number_functions_correctly_deployed': [sum_deployed],
        'functions_correctly_executed': [avg_executed],
        'number_functions_correctly_executed': [sum_executed],
        'avg_invocation_attempts': [avg_invocation_attempts],
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
