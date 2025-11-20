import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# TODO: Correct the names at the end
def make_vertical_bar_plot():
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



def make_horizontal_bar_plot():
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

        # Horizontal bars
        bars = ax.barh(models, valori, color=plot_colors, height=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if is_canonical:
            ax.axvline(x=canonical_value, color='red', linestyle='--', linewidth=2,
                       label=f'Value of canonical solution ({canonical_value})')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        ax.set_title(metric, fontsize=12, weight='bold')
        ax.set_xlabel(metric, fontsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        for bar in bars:
            xval = bar.get_width()
            ax.text(xval * 1.01, bar.get_y() + bar.get_height()/2.0, f'{xval:.2f}', ha='left', va='center', fontsize=8)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15, hspace=0.5, wspace=0.3)
    plt.savefig('../comparison_horizontal.png', dpi=300)
    plt.show()



def make_radar_plot():
    df_csv = pd.read_csv('../results.csv')
    avoid_models = ['deepseek-coder-v2_no_prompt', 'qwen2.5-coder_no_prompt', 'gemini-2.0-flash_no_prompt',
                    'gemini-2.5-pro_no_prompt', 'qwen2.5-coder:32b_no_prompt', 'canonical']
    df_csv = df_csv[~df_csv['model'].isin(avoid_models)].reset_index(drop=True)

    models = df_csv['model'].tolist()

    metrics = [
        'pass@1',
        'avg_generation_time (s)',
        'avg_tokens',
        'avg_execution_time (s)',
        'avg_cc_generation',
        'avg_cog_generation'
    ]

    name_metrics = [
        'pass@1',
        'avgerage\ngeneration\ntime (s)',
        'avgerage\ntokens',
        'avgerage\nexecution\ntime (s)',
        'avgerage\nCC generation',
        'avgerage\nCoG generation'
    ]

    df_filtered = df_csv[['model'] + metrics]

    # Data has to be normalized because each metric has it own scale
    df_normalized = df_filtered.drop('model', axis=1)
    lower_is_better = ['avg_generation_time (s)', 'avg_execution_time (s)', 'avg_tokens', 'avg_cc_generation',
                       'avg_cog_generation']
    new_min, new_max = 0.1, 1.0

    for metric in metrics:
        min_val, max_val = df_normalized[metric].min(), df_normalized[metric].max()
        if max_val - min_val == 0:
            df_normalized[metric] = new_max
            continue

        if metric in lower_is_better:
            df_normalized[metric] = new_min + ((max_val - df_normalized[metric]) / (max_val - min_val)) * (
                        new_max - new_min)
        else:
            df_normalized[metric] = new_min + ((df_normalized[metric] - min_val) / (max_val - min_val)) * (
                        new_max - new_min)

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    for i, model in enumerate(models):
        values = df_normalized.iloc[i].values.flatten().tolist()
        values += values[:1]
        color = color_map[model]

        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=model, marker='o')
        ax.fill(angles, values, color=color, alpha=0.2)


    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(name_metrics, size=14)
    ax.tick_params(axis='x', pad=30)

    yticks = np.arange(new_min, new_max + 0.1, 0.1)
    ax.set_yticks(yticks)

    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks], color="grey", size=10)

    ax.set_rlabel_position(30)
    ax.set_ylim(0, 1.1)

    plt.title('Comparing Performances', size=26, y=1.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=14)

    plt.tight_layout(pad=1.5)

    plt.savefig('../radar_comparison_final.png', dpi=300)
    plt.show()




#make_plot()
make_radar_plot()

