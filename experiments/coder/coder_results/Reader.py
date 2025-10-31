import pandas as pd

# Sostituisci con il percorso del tuo file
file_path = "deepseek-coder-v2.parquet"

try:
    # Leggi il file parquet
    df = pd.read_parquet(file_path)

    # Stampa le prime 10 righe del DataFrame
    print(df.head(133))

    # Se vuoi vedere l'intero contenuto (solo se il file è piccolo!)
    # print(df)

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")