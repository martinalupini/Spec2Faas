import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns



def make_pass1_horizontal_bar_plot_2(csv_path='../results.csv'):
    df_csv = pd.read_csv(csv_path)

    avoid_models = [
        'deepseek-coder-v2_no_prompt',
        'qwen2.5-coder_no_prompt',
        'gemini-2.0-flash_no_prompt',
        'gemini-2.5-pro_no_prompt',
        'qwen2.5-coder:32b_no_prompt',
        'canonical'
    ]

    df_plot = df_csv[~df_csv['model'].isin(avoid_models)].copy()

    unique_models = df_plot['model'].unique().tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))
    color_map = {model: color for model, color in zip(unique_models, colors)}

    df_plot = df_plot.sort_values(by='pass@1', ascending=True)

    models = df_plot['model'].tolist()
    values = df_plot['pass@1'].tolist()

    plot_colors = [color_map[m] for m in models]

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(models, values, color=plot_colors, height=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_yticklabels(models)

    ax.set_title('Metric Comparison: pass@1', fontsize=22, weight='bold', pad=20)
    ax.set_xlabel('pass@1', fontsize=22, labelpad=20, weight='bold')
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    max_val = max(values) if values else 1
    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + (max_val * 0.01), bar.get_y() + bar.get_height() / 2.0,
                f'{xval:.2f}', ha='left', va='center', fontsize=18, weight='bold')

    plt.tight_layout()
    plt.savefig('../pass1_comparison_horizontal.png', dpi=300)
    plt.show()

def make_performance_plots(csv_path='../results.csv'):
    df_csv = pd.read_csv(csv_path)

    canonical_df = df_csv[df_csv['model'] == 'canonical']

    avoid_models = [
        'deepseek-coder-v2_no_prompt',
        'qwen2.5-coder_no_prompt',
        'gemini-2.0-flash_no_prompt',
        'gemini-2.5-pro_no_prompt',
        'qwen2.5-coder:32b_no_prompt',
        'canonical'
    ]

    df_plot = df_csv[~df_csv['model'].isin(avoid_models)].copy()

    metrics = ['avg_generation_time (s)', 'avg_tokens']
    titles = ['Response Time (s)', 'Token Usage']

    models = df_plot['model'].tolist()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    for ax, metric, title in zip(axes, metrics, titles):
        is_canonical = False

        df_sorted = df_plot.sort_values(by=metric, ascending=True)
        current_models = df_sorted['model'].tolist()
        valori = df_sorted[metric].tolist()
        plot_colors = [color_map[m] for m in current_models]

        bars = ax.barh(current_models, valori, color=plot_colors, height=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if is_canonical and not canonical_df.empty:
            canonical_value = canonical_df[metric].iloc[0]
            ax.axvline(x=canonical_value, color='red', linestyle='--', linewidth=2,
                       label=f'Canonical: {canonical_value:.2f}s')
            ax.legend(loc='lower right', frameon=True)

        ax.set_title(title, fontsize=22, weight='bold')
        ax.set_xlabel(metric, fontsize=22, labelpad=20, weight='bold')
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='x', labelsize=18)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + (max(valori) * 0.01), bar.get_y() + bar.get_height() / 2.0,
                    f'{xval:.2f}', ha='left', va='center', fontsize=18, weight='bold')

    plt.tight_layout()
    plt.savefig('../performance_comparison.png', dpi=300)
    plt.show()


def make_radar_plot():
    df_csv = pd.read_csv('../results.csv')

    avoid_models = [
        'deepseek-coder-v2_no_prompt',
        'qwen2.5-coder_no_prompt',
        'gemini-2.0-flash_no_prompt',
        'gemini-2.5-pro_no_prompt',
        'qwen2.5-coder:32b_no_prompt',
        'canonical'
    ]
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
        'average\ngeneration\ntime (s)',
        'average\ntokens usage',
        'average\nexecution\ntime (s)',
        'average\ncyclomatic complexity',
        'average\ncognitive complexity'
    ]

    df_filtered = df_csv[['model'] + metrics]

    # Normalize metrics because each metric has its own scale
    df_normalized = df_filtered.drop('model', axis=1)

    lower_is_better = [
        'avg_generation_time (s)',
        'avg_execution_time (s)',
        'avg_tokens',
        'avg_cc_generation',
        'avg_cog_generation'
    ]

    new_min, new_max = 0.1, 1.0

    for metric in metrics:
        min_val, max_val = df_normalized[metric].min(), df_normalized[metric].max()

        if max_val - min_val == 0:
            df_normalized[metric] = new_max
            continue

        if metric in lower_is_better:
            df_normalized[metric] = new_min + ((max_val - df_normalized[metric]) / (max_val - min_val)) * (new_max - new_min)
        else:
            df_normalized[metric] = new_min + ((df_normalized[metric] - min_val) / (max_val - min_val)) * (new_max - new_min)

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    for i, model in enumerate(models):
        values = df_normalized.iloc[i].values.flatten().tolist()
        values += values[:1]
        color = color_map[model]

        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2,
            linestyle='solid',
            label=model,
            marker='o'
        )
        ax.fill(angles, values, color=color, alpha=0.2)

    # Bigger axis labels + more padding to avoid overlaps
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(name_metrics, fontsize=26, fontweight='bold')
    ax.tick_params(axis='x', pad=50)

    yticks = np.arange(new_min, new_max + 0.1, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [f'{tick:.1f}' for tick in yticks],
        fontsize=18,
        color="grey"
    )

    ax.set_rlabel_position(30)
    ax.set_ylim(0, 1.1)

    # Bigger title
    plt.title(
        'Comparing Performances for Coder Agent',
        fontsize=30,
        weight='bold',
        y=1.11
    )

    # Legend moved further outside so it won't overlap
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        fontsize=26
    )

    # IMPORTANT: avoid tight_layout because it compresses everything and can cause overlaps
    # plt.tight_layout(pad=1.5)

    plt.savefig('../radar_comparison_coder.png', dpi=300, bbox_inches='tight')
    plt.show()

def make_complexity_plots(csv_path='../results.csv'):
    df_csv = pd.read_csv(csv_path)
    canonical_df = df_csv[df_csv['model'] == 'canonical']

    avoid_models = [
        'deepseek-coder-v2_no_prompt',
        'qwen2.5-coder_no_prompt',
        'gemini-2.0-flash_no_prompt',
        'gemini-2.5-pro_no_prompt',
        'qwen2.5-coder:32b_no_prompt',
        'canonical'
    ]

    df_plot = df_csv[~df_csv['model'].isin(avoid_models)].copy()

    metrics = ['avg_cc_generation', 'avg_cog_generation']
    titles = ['Cyclomatic Complexity (CC)', 'Cognitive Complexity (CoGC)']

    models = df_plot['model'].tolist()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    for ax, metric, title in zip(axes, metrics, titles):
        df_sorted = df_plot.sort_values(by=metric, ascending=True)
        current_models = df_sorted['model'].tolist()
        valori = df_sorted[metric].tolist()
        plot_colors = [color_map[m] for m in current_models]

        bars = ax.barh(current_models, valori, color=plot_colors, height=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if not canonical_df.empty:
            canonical_value = canonical_df[metric].iloc[0]
            ax.axvline(x=canonical_value, color='red', linestyle='--', linewidth=2,
                       label=f'Canonical: {canonical_value:.2f}')
            ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), fancybox=True, shadow=True, fontsize=20)

        ax.set_title(title, fontsize=22, weight='bold')
        ax.set_xlabel(metric, fontsize=22, weight='bold', labelpad=40)
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='x', labelsize=18)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        max_val = max(valori) if valori else 1
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + (max_val * 0.01), bar.get_y() + bar.get_height() / 2.0,
                    f'{xval:.2f}', ha='left', va='center', fontsize=18, weight='bold')

    plt.subplots_adjust(
        left=0.15,
        right=0.98,
        wspace=0.6,
        bottom=0.2
    )
    plt.savefig('../complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_prompt_comparison(csv_path='../results.csv'):
    df_csv = pd.read_csv(csv_path)

    df_csv['is_no_prompt'] = df_csv['model'].str.contains('_no_prompt')
    df_csv['base_model'] = df_csv['model'].str.replace('_no_prompt', '')

    df_csv = df_csv[df_csv['base_model'] != 'canonical']

    models_with_both = df_csv.groupby('base_model').filter(lambda x: len(x) >= 2)['base_model'].unique()
    df_plot = df_csv[df_csv['base_model'].isin(models_with_both)].copy()

    order_df = df_plot[~df_plot['is_no_prompt']].sort_values(by='pass@1', ascending=True)
    sorted_models = order_df['base_model'].tolist()

    y = np.arange(len(sorted_models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(16, 8))

    prompt_values = [df_plot[(df_plot['base_model'] == m) & (~df_plot['is_no_prompt'])]['pass@1'].values[0] for m in
                     sorted_models]
    no_prompt_values = [df_plot[(df_plot['base_model'] == m) & (df_plot['is_no_prompt'])]['pass@1'].values[0] for m in
                        sorted_models]

    rects1 = ax.barh(y + height / 2, prompt_values, height, label='With Prompt', color='#1f77b4')
    rects2 = ax.barh(y - height / 2, no_prompt_values, height, label='No Prompt', color='#d62728')

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_models, fontsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlabel('pass@1 Score', fontsize=22, weight='bold', labelpad=40)
    ax.set_title('Pass@1 With Prompt (Blue) vs No Prompt (Red)', fontsize=22,
                 weight='bold')
    ax.legend(loc='lower right', fontsize=22, bbox_to_anchor=(1.02, 0.0))

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.text(width + 0.005, rect.get_y() + rect.get_height() / 2,
                    f'{width:.2f}', ha='left', va='center', fontsize=18, weight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('../prompt_comparison_pass1.png', dpi=300)
    plt.show()


def statistical_analysis():
    df = pd.read_csv('../results_deepseek.csv')

    metrics = [
        'pass@1',
        'avg_tokens',
        'avg_generation_time (s)',
        'avg_cc_generation',
        'avg_cog_generation'
    ]

    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(12, 8))

    colors = sns.color_palette("Set2", len(metrics))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        sns.boxplot(x=df[metric], ax=axes[i], color=color, width=0.5)

        axes[i].tick_params(axis='x', labelsize=18)

        axes[i].set_ylabel(metric, fontsize=22, fontweight='bold',
                           rotation=0, labelpad=30, ha='right', va='center')

        axes[i].set_xlabel('')
        if i == len(metrics) - 1:
            axes[i].set_xlabel('Value', fontsize=22, labelpad=20, weight='bold')

        axes[i].set_yticks([])

        axes[i].xaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(pad=3.0)

    plt.savefig('../statistical_analysis.png')
    plt.show()




#make_radar_plot()  #radar
#make_pass1_horizontal_bar_plot_2() #pass@1
#make_performance_plots()   #time and token
#make_complexity_plots()   #cc e cog
plot_prompt_comparison()  #with without prompt
#statistical_analysis()

