import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def make_vertical_bar_plot():
    df_csv = pd.read_csv('../results.csv')

    models = df_csv['model'].tolist()
    metrics = df_csv.columns.drop(['model', 'avg_invocation_attempts']).tolist()

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

        bars = ax.bar(models_plot, valori, color=plot_colors, width=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

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


make_vertical_bar_plot()