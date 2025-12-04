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

    title_text = f'Debbugging gain for coder {coder}'
    ax.set_title(title_text, fontsize=14, weight='bold')
    ax.set_ylabel('Number of corrected functions', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.01, f'{yval:.0f}', ha='center', va='bottom',
                fontsize=9, weight='bold')

    ax.set_ylim(bottom=0)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)

    save_path = f'../debugging_gain_{coder}.png'
    plt.savefig(save_path, dpi=300)

    plt.show()


make_debugging_gain_plot()