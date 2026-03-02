import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_horizontal_bar_plots(metrics, fig_lenght, title, labels):
    df_csv = pd.read_csv('../results.csv')

    all_possible_models = sorted(list(set(df_csv['model'].tolist() + ['qwen2.5-coder:32b'])))

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_possible_models)))
    color_map = {model: color for model, color in zip(all_possible_models, colors)}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_lenght, 6))

    for ax, metric, label in zip(axes, metrics, labels):
        df_sorted = df_csv[['model', metric]].sort_values(by=metric, ascending=True)

        models_plot = df_sorted['model'].tolist()
        values = df_sorted[metric].tolist()
        plot_colors = [color_map[m] for m in models_plot]

        bars = ax.barh(models_plot, values, color=plot_colors, height=0.6)

        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='x', labelsize=18)


        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(label, fontsize=22, weight='bold', labelpad=40)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max(values) * 0.01), bar.get_y() + bar.get_height() / 2,
                    f'{width}', va='center', fontsize=18, weight='bold')

    plt.tight_layout()
    plt.savefig(title, bbox_inches='tight')
    plt.show()


make_horizontal_bar_plots(['number_functions_correctly_deployed', 'number_functions_correctly_executed'], 20, '../comparison_horizontal_deployer.pdf', ["Number Functions Correctly Deployed", "Number Functions Executed Correctly"])
#make_horizontal_bar_plots(['avg_deployment_time (s)', 'avg_tokens'], 15, '../comparison_horizontal_deployer_performance.pdf')