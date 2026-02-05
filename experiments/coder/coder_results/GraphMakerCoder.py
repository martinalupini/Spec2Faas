import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def make_vertical_bar_plot():
    df_csv = pd.read_csv('../results.csv')

    canonical_df = df_csv[df_csv['model'] == 'canonical']

    avoid_models = ['deepseek-coder-v2_no_prompt', 'qwen2.5-coder_no_prompt', 'gemini-2.0-flash_no_prompt', 'gemini-2.5-pro_no_prompt', 'qwen2.5-coder:32b_no_prompt', 'canonical']
    df_csv = df_csv[~df_csv['model'].isin(avoid_models)]


    metrics = df_csv.columns.drop(['model', 'avg_execution_time (s)']).tolist()
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
        if metric == 'pass@1' or metric == 'number_func_generated' or metric == 'avg_tokens':
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

    metrics = df_csv.columns.drop(['model', 'avg_execution_time (s)']).tolist()
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
        if metric == 'pass@1' or metric == 'number_func_generated' or metric == 'avg_tokens':
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


def make_pass1_horizontal_bar_plot(csv_path='../results.csv'):
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

    df_plot = df_plot.sort_values(by='pass@1', ascending=True)

    models = df_plot['model'].tolist()
    values = df_plot['pass@1'].tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    bars = ax.barh(models, values, color=colors, height=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title('Metric Comparison: pass@1', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('pass@1', fontsize=20)
    ax.tick_params(axis='y', labelsize=10)

    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + (max(values) * 0.01), bar.get_y() + bar.get_height() / 2.0,
                f'{xval:.2f}', ha='left', va='center', fontsize=18, weight='bold')

    plt.tight_layout()
    plt.savefig('../pass1_comparison_horizontal.png', dpi=300)
    plt.show()


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
    ax.set_xlabel('pass@1', fontsize=22)
    ax.tick_params(axis='y', labelsize=20)

    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    max_val = max(values) if values else 1
    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + (max_val * 0.01), bar.get_y() + bar.get_height() / 2.0,
                f'{xval:.2f}', ha='left', va='center', fontsize=16, weight='bold')

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
        ax.set_xlabel(metric, fontsize=22)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + (max(valori) * 0.01), bar.get_y() + bar.get_height() / 2.0,
                    f'{xval:.2f}', ha='left', va='center', fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig('../performance_comparison.png', dpi=300)
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
        'avgerage\ntokens usage',
        'avgerage\nexecution\ntime (s)',
        'avgerage\nciclomatic complexity',
        'avgerage\ncognitive complexity'
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

    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    for i, model in enumerate(models):
        values = df_normalized.iloc[i].values.flatten().tolist()
        values += values[:1]
        color = color_map[model]

        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=model, marker='o')
        ax.fill(angles, values, color=color, alpha=0.2)


    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(name_metrics, size=20)
    ax.tick_params(axis='x', pad=30)

    yticks = np.arange(new_min, new_max + 0.1, 0.1)
    ax.set_yticks(yticks)

    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks], color="grey", size=10)

    ax.set_rlabel_position(30)
    ax.set_ylim(0, 1.1)

    plt.title('Comparing Performances for Coder Agent', size=22, y=1.1, weight='bold')

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=18)

    plt.tight_layout(pad=1.5)

    plt.savefig('../radar_comparison_coder.png', dpi=300)
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
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

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
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)

        ax.set_title(title, fontsize=22, weight='bold')
        ax.set_xlabel(metric, fontsize=22)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        max_val = max(valori) if valori else 1
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + (max_val * 0.01), bar.get_y() + bar.get_height() / 2.0,
                    f'{xval:.2f}', ha='left', va='center', fontsize=16, weight='bold')

    plt.subplots_adjust(bottom=0.2, wspace=0.4)
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

    fig, ax = plt.subplots(figsize=(14, 8))

    prompt_values = [df_plot[(df_plot['base_model'] == m) & (~df_plot['is_no_prompt'])]['pass@1'].values[0] for m in
                     sorted_models]
    no_prompt_values = [df_plot[(df_plot['base_model'] == m) & (df_plot['is_no_prompt'])]['pass@1'].values[0] for m in
                        sorted_models]

    rects1 = ax.barh(y + height / 2, prompt_values, height, label='With Prompt', color='#1f77b4')
    rects2 = ax.barh(y - height / 2, no_prompt_values, height, label='No Prompt', color='#d62728')

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_models, fontsize=20)

    ax.set_xlabel('pass@1 Score', fontsize=20)
    ax.set_title('Pass@1 With Prompt (Blue) vs No Prompt (Red)', fontsize=22,
                 weight='bold')
    ax.legend(loc='lower right', fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.text(width + 0.005, rect.get_y() + rect.get_height() / 2,
                    f'{width:.2f}', ha='left', va='center', fontsize=16, weight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('../prompt_comparison_pass1.png', dpi=300)
    plt.show()



#make_radar_plot()
#make_pass1_horizontal_bar_plot_2()
#make_performance_plots()
make_complexity_plots()
#plot_prompt_comparison()

