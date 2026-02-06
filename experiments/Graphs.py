import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def make_radar_plot():
    df_csv = pd.read_csv('results_final.csv')
    max_value = 165

    df_wide = df_csv.pivot_table(index='model', columns='metric', values='value', aggfunc='first')
    df_wide.reset_index(inplace=True)

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
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

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
    ax.set_xticklabels([])
    ax.tick_params(axis='x', pad=0)

    R_NORMAL = max_value * 1.05
    R_WIDE = max_value * 1.07


    for i, angle in enumerate(angles[:-1]):
        label = name_metrics[i]

        ha = 'center'
        va = 'center'
        r_label_position = R_NORMAL

        if i == 0:  # 0°: 'Code Generation'
            ha = 'left'
            r_label_position = R_WIDE

        elif i == 1:  # 90°: 'Debugging'
            va = 'bottom'

        elif i == 2:  # 180°: 'Test Generation'
            ha = 'right'
            r_label_position = R_WIDE

        elif i == 3:  # 270°: 'Deployment'
            va = 'top'

        ax.text(angle,
                r_label_position,
                label,
                size=20,
                horizontalalignment=ha,
                verticalalignment=va)
    yticks = np.arange(0, max_value, 25)
    ax.set_yticks(yticks)

    ax.set_yticklabels([f'{tick:.0f}' for tick in yticks], color="grey", size=10)

    ax.set_rlabel_position(30)
    ax.set_ylim(0, max_value)

    plt.title('Performance of models in different tasks', size=22, y=1.1, weight='bold')

    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=16)

    plt.tight_layout(pad=1.5)

    plt.savefig('radar_comparison_all.png', dpi=300)
    #plt.show()

make_radar_plot()