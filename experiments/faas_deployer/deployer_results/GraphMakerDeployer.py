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


def make_horizontal_bar_plots():
    df_csv = pd.read_csv('../results.csv')

    all_possible_models = sorted(list(set(df_csv['model'].tolist() + ['qwen2.5-coder:32b'])))

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_possible_models)))
    color_map = {model: color for model, color in zip(all_possible_models, colors)}

    #metrics = ['number_functions_correctly_deployed', 'number_functions_correctly_executed']
    metrics = ['avg_deployment_time (s)', 'avg_tokens']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    for ax, metric in zip(axes, metrics):
        df_sorted = df_csv[['model', metric]].sort_values(by=metric, ascending=True)

        models_plot = df_sorted['model'].tolist()
        values = df_sorted[metric].tolist()
        plot_colors = [color_map[m] for m in models_plot]

        bars = ax.barh(models_plot, values, color=plot_colors, height=0.6)

        ax.set_title(metric.replace('_', ' ').title(), fontsize=14, weight='bold')
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(metric.replace('_', ' ').title())

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max(values) * 0.01), bar.get_y() + bar.get_height() / 2,
                    f'{width}', va='center', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig('../comparison_horizontal_deployer_performance.png', dpi=300)
    plt.show()


#make_vertical_bar_plot()
make_horizontal_bar_plots()