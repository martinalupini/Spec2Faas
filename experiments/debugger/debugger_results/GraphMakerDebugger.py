import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from app.Utils import *
from matplotlib.gridspec import GridSpec

llm = get_config_data("../../config_test.yaml")
debugger = llm['debugger']
coder = llm['coder']


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


def plot_coder_comparison(
        csv_path="../results.csv",
        coders_list=("gemini-2.5-pro", "gemini-2.0-flash", "deepseek-coder-v2"),
):
    df_csv = pd.read_csv(csv_path)

    df_csv["debugging_gain"] = (
            df_csv["number_passed_after_debugging"]
            - df_csv["number_passed_after_generation"]
    )


    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 1:3])

    axes = [ax1, ax2, ax3]

    debuggers_in_csv = sorted(
        df_csv[df_csv["debugger"] != "no debugger"]["debugger"].unique()
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(debuggers_in_csv)))
    color_map = {model: color for model, color in zip(debuggers_in_csv, colors)}

    for i, coder_name in enumerate(coders_list):
        ax = axes[i]

        df_coder = df_csv[df_csv["coder"] == coder_name].copy()
        df_plot = df_coder[df_coder["debugger"] != "no debugger"].copy()

        if df_plot.empty:
            ax.text(0.5, 0.5, f"No data for\n{coder_name}", ha="center", va="center")
            ax.set_title(f"Coder: {coder_name}", fontsize=22, weight="bold", pad=8)
            continue

        df_plot = df_plot.sort_values(by="debugging_gain", ascending=True)

        debuggers_y = df_plot["debugger"].tolist()
        gains_x = df_plot["debugging_gain"].tolist()
        bar_colors = [color_map.get(dbg, "gray") for dbg in debuggers_y]

        bars = ax.barh(debuggers_y, gains_x, color=bar_colors, height=0.6)


        local_max = max(gains_x) if gains_x else 1
        x_limit = local_max + (local_max * 0.2) + 1
        ax.set_xlim(0, x_limit)

        if local_max <= 5:
            step = 1
        elif local_max <= 20:
            step = 2
        else:
            step = 5
        ax.set_xticks(np.arange(0, local_max + step, step))

        ax.grid(axis="x", linestyle="--", alpha=0.4)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(f"Coder: {coder_name}", fontsize=22, weight="bold", pad=20)
        ax.set_xlabel("Corrected Functions", fontsize=22, weight="bold", labelpad=15)

        ax.tick_params(axis="y", labelsize=22)
        ax.tick_params(axis="x", labelsize=18)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (x_limit * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                ha="left",
                va="center",
                weight="bold",
                fontsize=18,
            )

    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(hspace=0.5)

    output_name = "../comparison_gain.pdf"
    plt.savefig(output_name, bbox_inches="tight")
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


    color_base = '#a6cee3'
    color_gain = '#27ae60'

    bar_width = 0.65

    p1 = ax.bar(coders, initial_passed, bar_width, label='Initial Generation',
                color=color_base, zorder=3, alpha=0.9)

    p2 = ax.bar(coders, debugging_gain, bar_width, bottom=initial_passed,
                label='Debugging Gain', color=color_gain, zorder=3, alpha=0.9)

    ax.bar_label(p1, label_type='center', color='#2c3e50', weight='bold', fontsize=22)

    ax.bar_label(p2, label_type='edge', padding=3, weight='bold', fontsize=22, color='#2c3e50')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)

    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.15)

    ax.set_title(f'Performance Analysis Debugger: {debugger_name}', fontsize=22, weight='bold', pad=25)
    ax.set_ylabel('Number of Functions Corrected', fontsize=22, labelpad=10, weight='bold')
    ax.set_xlabel('Coder Agent', fontsize=22, weight='bold', labelpad=40)

    ax.tick_params(axis='x', rotation=0, labelsize=22)
    ax.tick_params(axis='y', labelsize=18)


    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
              ncol=2, frameon=False, fontsize=22)

    plt.tight_layout()

    output_path = f"../performance_debugger_{debugger_name.replace('.', '_')}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()


#make_debugging_gain_plot()
plot_coder_comparison()
#plot_debugger_performance()