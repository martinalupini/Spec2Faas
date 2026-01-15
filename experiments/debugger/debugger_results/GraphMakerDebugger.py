import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from app.Utils import *

llm = get_config_data("../../config_test.yaml")
debugger = llm['debugger']
coder = llm['coder']


def make_vertical_bar_plot():
    df_csv = pd.read_csv('../results.csv')
    df_csv = df_csv[df_csv['coder'] == coder].copy()
    models = df_csv['debugger'].tolist()
    metrics = df_csv.columns.drop(['coder', 'debugger','passed_after_generation', 'number_passed_after_generation', 'avg_generation_time (s)', 'avg_debugging_tokens']).tolist()
    df_csv_no_canonical = df_csv[df_csv['debugger'] != "no debugger"].copy()
    models_no_canonical = df_csv_no_canonical['debugger'].tolist()


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

        if metric == 'average_debugging_time (s)' or metric == 'avg_attempts_debugging':
            df_value = df_csv_no_canonical.copy()
            models_plot = models_no_canonical
        else:
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
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.01, f'{yval:.2f}', ha='center', va='bottom',
                    fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15, hspace=0.5, wspace=0.25)

    plt.savefig('../comparison_'+ coder+'.png', dpi=300)

    plt.show()


def make_vertical_bar_plot_2():
    df_csv = pd.read_csv('../results.csv')
    df_csv = df_csv[df_csv['coder'] == coder].copy()
    models = df_csv['debugger'].tolist()
    metrics = df_csv.columns.drop(['coder', 'debugger','passed_after_generation', 'number_passed_after_generation', 'avg_generation_time (s)', 'avg_total_tokens', 'average_total_time (s)', 'passed_after_debugging']).tolist()
    df_csv_no_canonical = df_csv[df_csv['debugger'] != "no debugger"].copy()
    models_no_canonical = df_csv_no_canonical['debugger'].tolist()


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

        if metric == 'average_debugging_time (s)' or metric == 'avg_attempts_debugging' or metric == 'avg_debugging_tokens':
            df_value = df_csv_no_canonical.copy()
            models_plot = models_no_canonical
        else:
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
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.01, f'{yval:.2f}', ha='center', va='bottom',
                    fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15, hspace=0.5, wspace=0.25)

    plt.savefig('../comparison_2_'+ coder+'.png', dpi=300)

    plt.show()


def make_debugging_gain_plot():

    df_csv = pd.read_csv('../results.csv')

    df_csv = df_csv[df_csv['coder'] == coder].copy()

    df_csv_no_canonical = df_csv[df_csv['debugger'] != "no debugger"].copy()
    models_plot = df_csv_no_canonical['debugger'].tolist()

    df_csv_no_canonical['debugging_gain'] = (
            df_csv_no_canonical['number_passed_after_debugging'] -
            df_csv_no_canonical['number_passed_after_generation']
    )

    metric = 'debugging_gain'
    valori = df_csv_no_canonical[metric]


    fig, ax = plt.subplots(figsize=(8, 6))


    models = df_csv['debugger'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}
    plot_colors = [color_map[model] for model in models_plot]

    bars = ax.bar(models_plot, valori, color=plot_colors, width=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    title_text = f'Debugging gain for coder {coder}'
    ax.set_title(title_text, fontsize=14, weight='bold')
    ax.set_ylabel('Number of corrected functions', fontsize=16)
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.01, f'{yval:.0f}', ha='center', va='bottom',
                fontsize=11, weight='bold')

    ax.set_ylim(bottom=0)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)

    save_path = f'../debugging_gain_{coder}.png'
    plt.savefig(save_path, dpi=300)

    plt.show()



def plot_coder_comparison(csv_path='../results.csv', coders_list=['gemini-2.5-pro', 'gemini-2.0-flash', 'deepseek-coder-v2']):

    df_csv = pd.read_csv(csv_path)

    num_coders = len(coders_list)
    fig, axes = plt.subplots(1, num_coders, figsize=(22, 6), sharex=True)

    if num_coders == 1:
        axes = [axes]

    debuggers_in_csv = set(df_csv[df_csv['debugger'] != "no debugger"]['debugger'].unique())


    all_potential_debuggers = sorted(list(debuggers_in_csv | {"qwen2.5-coder"}))

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_potential_debuggers)))
    color_map = {model: color for model, color in zip(all_potential_debuggers, colors)}

    for i, coder_name in enumerate(coders_list):
        ax = axes[i]

        df_coder = df_csv[df_csv['coder'] == coder_name].copy()
        df_plot = df_coder[df_coder['debugger'] != "no debugger"].copy()

        if df_plot.empty:
            ax.text(0.5, 0.5, f"No data for\n{coder_name}", ha='center', va='center')
            ax.set_title(f"Coder: {coder_name}")
            continue

        df_plot['debugging_gain'] = (
                df_plot['number_passed_after_debugging'] -
                df_plot['number_passed_after_generation']
        )

        df_plot = df_plot.sort_values(by='debugging_gain', ascending=True)

        #df_plot = df_plot.sort_values('debugger', ascending=False)

        debuggers_y = df_plot['debugger'].tolist()
        gains_x = df_plot['debugging_gain'].tolist()
        bar_colors = [color_map[dbg] for dbg in debuggers_y]

        bars = ax.barh(debuggers_y, gains_x, color=bar_colors, height=0.6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(f'Coder: {coder_name}', fontsize=16, weight='bold', pad=15)

        if i == 0:
            ax.set_ylabel('Debugger Model', fontsize=12, weight='semibold')
        else:

            ax.tick_params(axis='y', length=0)

        ax.set_xlabel('Functions Corrected', fontsize=11)
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{int(width)}', ha='left', va='center',
                    weight='bold', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)

    output_name = "../comparison_gain.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.show()




def plot_debugger_performance(csv_path='../results.csv', debugger_name='gemini-2.5-pro'):
    df_csv = pd.read_csv(csv_path)

    df_filtered = df_csv[df_csv['debugger'] == debugger_name].copy()

    df_filtered['gain'] = (
            df_filtered['number_passed_after_debugging'] -
            df_filtered['number_passed_after_generation']
    )

    df_filtered = df_filtered.sort_values('number_passed_after_debugging', ascending=False)

    coders = df_filtered['coder'].tolist()
    initial_passed = df_filtered['number_passed_after_generation'].tolist()
    debugging_gain = df_filtered['gain'].tolist()

    fig, ax = plt.subplots(figsize=(12, 8))

    #color_base = '#34495e'  # Blu scuro (Generazione)
    color_base = '#a6cee3'
    color_gain = '#27ae60'  # Verde (Gain)
    #color_gain = '#a6cee3'

    bar_width = 0.65

    p1 = ax.bar(coders, initial_passed, bar_width, label='Initial Generation',
                color=color_base, zorder=3, alpha=0.9)

    p2 = ax.bar(coders, debugging_gain, bar_width, bottom=initial_passed,
                label='Debugging Gain', color=color_gain, zorder=3, alpha=0.9)

    ax.bar_label(p1, label_type='center', color='#2c3e50', weight='bold', fontsize=10)

    ax.bar_label(p2, label_type='edge', padding=3, weight='bold', fontsize=11, color='#2c3e50')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)

    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.15)

    ax.set_title(f'Performance Analysis Debugger: {debugger_name}', fontsize=18, weight='bold', pad=25)
    ax.set_ylabel('Number of Functions Corrected', fontsize=13, labelpad=10)
    ax.set_xlabel('Coder Agent', fontsize=13)

    ax.tick_params(axis='x', rotation=0, labelsize=11)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()

    output_path = f"../performance_debugger_{debugger_name.replace('.', '_')}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()


#make_debugging_gain_plot()
#make_vertical_bar_plot_2()
plot_coder_comparison()
#plot_debugger_performance()