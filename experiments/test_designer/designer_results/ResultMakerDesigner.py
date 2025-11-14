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
    avg_coverage = df['coverage'].mean()


    print(f"Passed: {passed:.4f}")
    print(f"Num passed: {num_passed:.4f}")
    print(f"\nAverage Generation Time: {avg_generation_time:.4f}")
    print(f"\nAverage Tokens: {avg_tokens:.4f}")
    print(f"Average Execution Time: {avg_execution_time_generation:.4f}")
    print(f"\nAverage Coverage: {avg_coverage:.4f}")

    results_data = {
        'model': [designer],
        'passed': [passed],
        'num_passed': [num_passed],
        'avg_test_generation_time': [avg_generation_time],
        'avg_tokens': [avg_tokens],
        'avg_test_execution_time': [avg_execution_time_generation],
        'avg_coverage': [avg_coverage]
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


write_csv()
#make_plot()