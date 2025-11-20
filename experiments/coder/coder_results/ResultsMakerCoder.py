import numpy as np
import pandas as pd
import os
from app.Utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

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


def write_csv():
    pass_at_1 = df['passed'].mean()
    avg_generation_time = df['generation_time'].mean()
    avg_tokens = df['tokens'].mean()
    avg_execution_time_generation = df['execution_time'].mean()
    avg_execution_time_canonical = df['execution_time_canonical'].mean()
    avg_cc_generation = df['CC_generation'].mean()
    avg_cc_canonical = df['CC_canonical'].mean()
    avg_cog_generation = df['CoG_generation'].mean()
    avg_cog_canonical = df['CoG_canonical'].mean()

    print(f"pass@1: {pass_at_1:.4f}")
    print(f"\nAverage Generation Time: {avg_generation_time:.4f}")
    print(f"\nAverage Tokens: {avg_tokens:.4f}")
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
        'avg_generation_time (s)': [avg_generation_time],
        'avg_tokens': [avg_tokens],
        'avg_execution_time (s)': [avg_execution_time_generation],
        'avg_cc_generation': [avg_cc_generation],
        'avg_cog_generation': [avg_cog_generation],
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

    canonical_df = df_csv[df_csv['model'] == 'canonical']

    avoid_models = ['deepseek-coder-v2_no_prompt', 'qwen2.5-coder_no_prompt', 'gemini-2.0-flash_no_prompt', 'gemini-2.5-pro_no_prompt', 'qwen2.5-coder:32b_no_prompt', 'canonical']
    df_csv = df_csv[~df_csv['model'].isin(avoid_models)]


    metrics = df_csv.columns.drop(['model']).tolist()
    models = df_csv['model'].tolist()

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
        if metric == 'pass@1' or metric == 'avg_generation_time (s)' or metric == 'avg_tokens':
            is_canonical = False
        else:
            is_canonical = True
            canonical_value = canonical_df[metric].iloc[0]

        valori = df_csv[metric]

        plot_colors = [color_map[model] for model in models]

        bars = ax.bar(models, valori, color=plot_colors, width=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if is_canonical:
            ax.axhline(y=canonical_value, color='red', linestyle='--', linewidth=2,
                       label=f'Value of canonical solution ({canonical_value})')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=1)

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