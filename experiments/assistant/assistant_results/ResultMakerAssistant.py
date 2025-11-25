import pandas as pd
from app.Utils import *

llm = get_config_data("../../config_test.yaml")
assistant = llm['assistant']


file_path = assistant + ".parquet"
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
    deployment = ((df['actual_type'] == 'deployment') & (df['type'] == df['actual_type'])).mean()
    translation = ((df['actual_type'] == 'translation') & (df['type'] == df['actual_type'])).mean()
    num_deployment = ((df['actual_type'] == 'deployment') & (df['type'] == df['actual_type'])).sum()
    num_translations = ((df['actual_type'] == 'translation') & (df['type'] == df['actual_type'])).sum()
    num_translation_as_other = ((df['actual_type'] == 'translation') & (df['type'] == 'other')).sum()
    num_translation_as_deployment = ((df['actual_type'] == 'translation') & (df['type'] == 'deployment')).sum()
    num_deployment_as_other = ((df['actual_type'] == 'deployment') & (df['type'] == 'other')).sum()
    num_deployment_as_translation = ((df['actual_type'] == 'deployment') & (df['type'] == 'translation')).sum()

    avg_time = df['time'].mean()
    avg_tokens = df['tokens'].mean()


    results_data = {
        'model': [assistant],
        'number_correct_deployment': [num_deployment],
        'number_correct_translation': [num_translations],
        'number_translation_as_other': [num_translation_as_other],
        'number_translation_as_deployment': [num_translation_as_deployment],
        'number_deployment_as_other': [num_deployment_as_other],
        'number_deployment_as_translation': [num_deployment_as_translation],
        'avg_tokens': [avg_tokens],
        'avg_time': [avg_time],
        'correct_deployment': deployment,
        'correct_translation': translation
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