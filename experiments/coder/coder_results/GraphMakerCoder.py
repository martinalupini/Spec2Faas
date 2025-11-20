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


def make_plot():
    # Carica i dati e filtra i modelli da non visualizzare
    df_csv = pd.read_csv('../results.csv')
    canonical_df = df_csv[df_csv['model'] == 'canonical']
    avoid_models = ['deepseek-coder-v2_no_prompt', 'qwen2.5-coder_no_prompt', 'gemini-2.0-flash_no_prompt', 'gemini-2.5-pro_no_prompt', 'qwen2.5-coder:32b_no_prompt', 'canonical']
    df_csv = df_csv[~df_csv['model'].isin(avoid_models)]

    # Estrae metriche e modelli
    metrics = df_csv.columns.drop(['model']).tolist()
    models = df_csv['model'].tolist()

    # Imposta la griglia per i subplot
    n_metrics = len(metrics)
    cols = min(n_metrics, 3)
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    # Nasconde gli assi extra
    for ax in axes.flatten()[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle('', fontsize=24, weight='bold')

    # Crea una mappa di colori per i modelli
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    # Itera su ogni metrica per creare un grafico
    for ax, metric in zip(axes.flatten(), metrics):
        if metric == 'pass@1' or metric == 'avg_generation_time (s)' or metric == 'avg_tokens':
            is_canonical = False
        else:
            is_canonical = True
            canonical_value = canonical_df[metric].iloc[0]

        valori = df_csv[metric]
        plot_colors = [color_map[model] for model in models]

        # --- MODIFICHE PRINCIPALI ---
        # 1. Usa barh per barre orizzontali e 'height' per renderle più sottili
        bars = ax.barh(models, valori, color=plot_colors, height=0.6)

        # 2. Rimuove i bordi del grafico
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Se esiste un valore canonico, disegna una linea verticale
        if is_canonical:
            ax.axvline(x=canonical_value, color='red', linestyle='--', linewidth=2,
                       label=f'Value of canonical solution ({canonical_value})')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        # Imposta titoli ed etichette
        ax.set_title(metric, fontsize=12, weight='bold')
        ax.set_xlabel(metric, fontsize=10) # Etichetta sull'asse x
        ax.tick_params(axis='y', rotation=0, labelsize=8) # Rimuove la rotazione sull'asse y
        ax.xaxis.grid(True, linestyle='--', alpha=0.6) # Griglia sull'asse x

        # Aggiunge le etichette dei valori alla fine delle barre
        for bar in bars:
            xval = bar.get_width()
            # Posiziona il testo leggermente dopo la fine della barra
            ax.text(xval * 1.01, bar.get_y() + bar.get_height()/2.0, f'{xval:.2f}', ha='left', va='center', fontsize=8)

    # Regola il layout e salva il grafico
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15, hspace=0.5, wspace=0.3)
    plt.savefig('../comparison_horizontal.png', dpi=300)
    plt.show()



make_plot()