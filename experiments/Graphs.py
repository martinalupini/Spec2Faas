import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def make_radar_plot():
    df_csv = pd.read_csv('results_final.csv')
    max_value = 165

    df_wide = df_csv.pivot_table(index='model', columns='metric', values='value', aggfunc='first')
    df_wide.reset_index(inplace=True)

    print(df_wide)
    df_wide = df_wide.drop_duplicates()

    models = df_wide['model'].tolist()

    metrics = [
        'code generation',
        'debugging',
        'test generation',
        'deployment'
    ]

    name_metrics = [
        'Code Generation',
        'Debugging',
        'Test Generation',
        'Deployment'
    ]

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Aggiungi il primo angolo alla fine per chiudere il cerchio

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    colors = plt.cm.get_cmap('Set1', len(models))
    color_map = {model: colors(i) for i, model in enumerate(models)}

    for i, model in enumerate(models):
        values = df_wide.loc[df_wide['model'] == model, metrics].values.flatten().tolist()

        values = [float(v) for v in values]

        values += values[:1]

        color = color_map[model]

        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=model, marker='o')
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(name_metrics, size=14)
    ax.tick_params(axis='x', pad=30)

    yticks = np.arange(0, max_value, 16)
    ax.set_yticks(yticks)

    ax.set_yticklabels([f'{tick:.0f}' for tick in yticks], color="grey", size=10)

    ax.set_rlabel_position(30)
    ax.set_ylim(0, max_value)

    # plt.title('Comparison among models', size=26, y=1.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=14)

    plt.tight_layout(pad=1.5)

    plt.savefig('radar_comparison.png', dpi=300)
    plt.show()

make_radar_plot()