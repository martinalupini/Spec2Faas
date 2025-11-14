import numpy as np
import pandas as pd
import os
from app.Utils import *
import matplotlib.pyplot as plt
import math

llm = get_config_data("../../config_test.yaml")
deployer = llm['faas_deployer']
if llm['coder_prompt'] == "Yes":
    prompt = True
else:
    prompt = False

if not prompt:
    deployer = deployer + "_no_prompt"

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



def make_plot():
    df_csv = pd.read_csv('../results.csv')

    models = df_csv['model'].tolist()
    metrics = df_csv.columns.drop(['model']).tolist()

    n_metrics = len(metrics)
    cols = min(n_metrics, 3)
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes.flatten()[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle('', fontsize=24, weight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    for ax, metric in zip(axes.flatten(), metrics):

        df_value = df_csv.copy()
        models_plot = models
        valori = df_value[metric]

        plot_colors = [color_map[model] for model in models_plot]

        bars = ax.bar(models_plot, valori, color=plot_colors)

        ax.set_title(metric, fontsize=12, weight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15, hspace=0.5, wspace=0.25)

    plt.savefig('../comparison.png', dpi=300)

    plt.show()


#write_csv()
make_plot()