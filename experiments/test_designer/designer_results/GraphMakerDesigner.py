import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    ax.set_xlabel(display_name, fontsize=22, weight='bold', labelpad=40)
    ax.set_yticks(range(len(models_sorted)))
    ax.tick_params(axis='x', labelsize=18)
    ax.set_yticklabels(models_sorted, fontsize=22)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', ha='left', va='center', fontsize=18, weight='bold')

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

        df_sorted = df_csv.sort_values(by=metric, ascending=True)
        current_models = df_sorted['model'].tolist()
        valori = df_sorted[metric].tolist()
        plot_colors = [color_map[m] for m in current_models]

        bars = ax.barh(current_models, valori, color=plot_colors, height=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(title, fontsize=22, weight='bold')
        ax.set_xlabel(metric, fontsize=22, weight='bold', labelpad=40)
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='x', labelsize=18)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + (max(valori) * 0.01), bar.get_y() + bar.get_height() / 2.0,
                    f'{xval:.2f}', ha='left', va='center', fontsize=18, weight='bold')

    plt.tight_layout()
    plt.savefig('../performance_comparison_designer.png', dpi=300)
    plt.show()



#make_metric_plot('passed', 'pass@1')
make_metric_plot('avg_coverage (%)', 'average test coverage (%)')
#make_performance_plots()