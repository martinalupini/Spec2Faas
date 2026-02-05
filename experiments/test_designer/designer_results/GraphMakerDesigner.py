import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def make_vertical_bar_plot():
    df_csv = pd.read_csv('../results.csv')

    models = df_csv['model'].tolist()
    # To be coherent with other graphs
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


def make_metric_plot(metric, display_name=None):
    if display_name is None:
        display_name = metric

    try:
        df_csv = pd.read_csv('../results.csv')
    except FileNotFoundError:
        print("Error: The file '../results.csv' was not found.")
        return

    if metric not in df_csv.columns:
        print(f"Error: The metric '{metric}' is not present in the CSV columns.")
        print(f"Available metrics: {df_csv.columns.tolist()}")
        return

    all_models = df_csv['model'].unique()
    colors_list = plt.cm.viridis(np.linspace(0, 1, len(all_models)))
    color_map = {model: color for model, color in zip(all_models, colors_list)}

    df_sorted = df_csv.sort_values(by=metric, ascending=True)

    models_sorted = df_sorted['model'].tolist()
    values_sorted = df_sorted[metric].values

    plot_colors = [color_map[m] for m in models_sorted]

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(models_sorted, values_sorted, color=plot_colors, height=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f'Model Comparison: {display_name}', fontsize=22, weight='bold')
    ax.set_xlabel(display_name, fontsize=20)
    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels(models_sorted, fontsize=20)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', ha='left', va='center', fontsize=16, weight='bold')

    plt.tight_layout()

    save_name = display_name.lower().replace(' ', '_')
    plt.savefig(f'../comparison_{save_name}_designer.png', dpi=300)
    plt.show()


def make_performance_plots(csv_path='../results.csv'):
    df_csv = pd.read_csv(csv_path)


    metrics = ['avg_test_generation_time (s)', 'avg_tokens']
    titles = ['Response Time (s)', 'Token Usage']

    models = df_csv['model'].tolist()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    for ax, metric, title in zip(axes, metrics, titles):
        is_canonical = False

        df_sorted = df_csv.sort_values(by=metric, ascending=True)
        current_models = df_sorted['model'].tolist()
        valori = df_sorted[metric].tolist()
        plot_colors = [color_map[m] for m in current_models]

        bars = ax.barh(current_models, valori, color=plot_colors, height=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(title, fontsize=22, weight='bold')
        ax.set_xlabel(metric, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + (max(valori) * 0.01), bar.get_y() + bar.get_height() / 2.0,
                    f'{xval:.2f}', ha='left', va='center', fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig('../performance_comparison_designer.png', dpi=300)
    plt.show()


#make_vertical_bar_plot()
#make_metric_plot('passed', 'pass@1')
make_metric_plot('avg_coverage (%)', 'average test coverage (%)')
#make_performance_plots()