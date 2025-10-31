import pandas as pd
import os

file_path = "deepseek-coder-v2.parquet"
csv_file = "../results.csv"

try:
    df = pd.read_parquet(file_path)

    #print(df.head(10))

    #print(df.info())

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")

pass_at_1 = df['passed'].mean()
avg_execution_time_generation = df['execution time'].mean()
avg_execution_time_canonical = df['execution time canonical'].mean()
avg_cc_generation = df['CC generation'].mean()
avg_cc_canonical = df['CC canonical'].mean()
avg_cog_generation = df['CoG generation'].mean()
avg_cog_canonical = df['CoG canonical'].mean()

print(f"pass@1: {pass_at_1:.4f}")
print("\n--- Average Execution Time ---")
print(f"Generated function: {avg_execution_time_generation:.4f}")
print(f"Canonical function: {avg_execution_time_canonical:.4f}")
print("\n--- Average Cyclomatic Complexity (CC) ---")
print(f"Generated function: {avg_cc_generation:.4f}")
print(f"Canonical function: {avg_cc_canonical:.4f}")
print("\n--- Average Cognitive Complexity (CoG) ---")
print(f"Generated function: {avg_cog_generation:.4f}")
print(f"Canonical function: {avg_cog_canonical:.4f}")


results_data = {
    'pass@1': [pass_at_1],
    'avg_execution_time_generation': [avg_execution_time_generation],
    'avg_execution_time_canonical': [avg_execution_time_canonical],
    'avg_cc_generation': [avg_cc_generation],
    'avg_cc_canonical': [avg_cc_canonical],
    'avg_cog_generation': [avg_cog_generation],
    'avg_cog_canonical': [avg_cog_canonical],
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