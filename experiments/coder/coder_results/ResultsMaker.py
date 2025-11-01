import numpy as np
import pandas as pd
import os
from app.Utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

llm = get_config_data("../../../config.yaml")
coder = llm['coder']

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


def write_csv():
    pass_at_1 = df['passed'].mean()
    avg_generation_time = df['generation time'].mean()
    avg_execution_time_generation = df['execution time'].mean()
    avg_execution_time_canonical = df['execution time canonical'].mean()
    avg_cc_generation = df['CC generation'].mean()
    avg_cc_canonical = df['CC canonical'].mean()
    avg_cog_generation = df['CoG generation'].mean()
    avg_cog_canonical = df['CoG canonical'].mean()

    print(f"pass@1: {pass_at_1:.4f}")
    print(f"\nAverage Generation Time: {avg_generation_time:.4f}")
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
        'model': [coder],
        'pass@1': [pass_at_1],
        'generation_time': [avg_generation_time],
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



def make_plot():
    df_csv = pd.read_csv('../results.csv')

    avoid_model = 'deepseek-coder-v2_no_prompt'
    df_csv = df_csv[df_csv['model'] != avoid_model].copy()

    models = df_csv['model'].tolist()
    metrics = df_csv.columns.drop('model').tolist()

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 12))

    fig.suptitle('', fontsize=24, weight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    for ax, metrica in zip(axes.flatten(), metrics):
        valori = df_csv[metrica]

        bars = ax.bar(models, valori, color=colors)

        ax.set_title(metrica, fontsize=12, weight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

    # Optional: add legend (but it does not fit)
    #legend_patches = [mpatches.Patch(color=color, label=model) for color, model in zip(colors, models)]
    #fig.legend(handles=legend_patches, loc='lower center', ncol=len(models), fontsize=12, title="Models")

    #plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15, hspace=0.5, wspace=0.25)

    plt.savefig('../comparison.png', dpi=300)

    plt.show()


#write_csv()
make_plot()