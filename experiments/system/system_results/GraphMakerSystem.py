import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from app.Utils import *
import numpy as np

config = get_config_data_full("../../config_test.yaml")
experiment = config["experiment_number"]


def create_detailed_sankey_diagram(experiment):

    try:
        df_csv = pd.read_csv('../results.csv')
    except FileNotFoundError:
        print("Error File not found: Please ensure '../results.csv' exists.")
        return

    df_experiment = df_csv[df_csv['experiment_number'] == experiment].copy()

    if df_experiment.empty:
        print(f"No data for experiment {experiment}")
        return

    for row in df_experiment.itertuples(index=False):
        dir_name = f"experiment_{row.experiment_number}/"
        os.makedirs(dir_name, exist_ok=True)

        execution_loss = row.num_deployed - row.num_correctly_executed
        total_items = 164

        initial_labels = [
            "Total",  # 0
            "Debugged",  # 1
            "Corrected",  # 2
            "Not Corrected",  # 3
            "Not Debugged",  # 4
            "Deployed",  # 5
            "Not Deployed",  # 6
            "Correctly Executed", # 7
            "Execution Failed" # 8
        ]

        link_values = [
            row.num_debugged, # 0 -> 1 (Total -> Debugged)
            total_items - row.num_debugged, # 0 -> 4 (Total -> Not Debugged)
            row.num_generated_and_debugged, # 1 -> 2 (Debugged -> Corrected)
            row.num_debugged - row.num_generated_and_debugged, # 1 -> 3 (Debugged -> Not Corrected)
            row.num_generated_debugged_deployed, # 2 -> 5 (Corrected -> Deployed)
            row.num_generated_debugged_not_deployed, # 2 -> 6 (Corrected -> Not Deployed)
            row.num_debugged - row.num_generated_and_debugged, # 3 -> 6 (Not Corrected -> Not Deployed) - *NOTE: This seems wrong in original code; it should be num_not_corrected_not_deployed*
            row.num_generated_not_debugged_deployed, # 4 -> 5 (Not Debugged -> Deployed)
            row.num_generated_not_debugged_not_deployed, # 4 -> 6 (Not Debugged -> Not Deployed)
            row.num_correctly_executed, # 5 -> 7 (Deployed -> Correctly Executed)
            execution_loss # 5 -> 8 (Deployed -> Execution Failed)
        ]

        source_nodes = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5]
        target_nodes = [1, 4, 2, 3, 5, 6, 6, 5, 6, 7, 8]

        node_totals = [0] * len(initial_labels)
        node_totals[0] = total_items # Total node has the fixed total value

        for src, tgt, val in zip(source_nodes, target_nodes, link_values):

            if tgt != 0:
                if val >= 0:
                    node_totals[tgt] += val
                else:
                    print(f"Warning: Negative flow value detected for link {initial_labels[src]} -> {initial_labels[tgt]}. Assuming 0 for calculation.")


        updated_labels = []
        for i, label in enumerate(initial_labels):
            updated_labels.append(f"{label} ({node_totals[i]})")


        nodes = dict(
            pad=100,
            thickness=18,
            line=dict(color="black", width=0.5),
            label=updated_labels,
            color=[
                "grey",  # Total
                "blue",  # Debugged
                "lightblue", #Corrected
                "orange",  # Not Corrected
                "magenta",  # Not Debugged
                "purple", #Deployed
                "red",  # Not Deployed
                "green",  # Correctly Executed
                "darkred"  # Execution Failed
            ],
            x=[0.01, 0.3, 0.5, 0.5, 0.3, 0.7, 0.7, 0.99, 0.99],
            y=[0.5, 0.75, 0.7, 0.93, 0.3, 0.35, 0.93, 0.3, 0.8],
            align="left"
        )

        alpha = 0.4
        link_colors = []
        for src_index in source_nodes:
            color_name = nodes['color'][src_index]
            r, g, b, _ = mcolors.to_rgba(color_name)
            color_string = f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})'
            link_colors.append(color_string)

        # Links between nodes
        links = dict(
            source=source_nodes,
            target=target_nodes,
            value=link_values,
            color=link_colors
        )

        fig = go.Figure(data=[go.Sankey(node=nodes, link=links, arrangement='freeform')])

        fig.update_layout(
            font=dict(
                size=28,
                color="black",
                weight="bold"
            )
        )

        fig.show()

        try:
            output_path = os.path.join(dir_name, 'detailed_sankey_flow.png')
            fig.write_image(output_path, width=2000, height=800, scale=2)
            print(f"Diagram saved in {output_path}")
        except Exception as e:
            print(f"Error in saving the diagram: {e}")






def analyze_and_visualize_results():
    try:
        df = pd.read_csv("../results.csv")
    except FileNotFoundError:
        return "Error: File not found."

    last_row = df.iloc[-1]
    deployed = last_row['num_deployed']
    executed = last_row['num_correctly_executed']

    labels = ['Deployed', 'Executed']
    values = [deployed, executed]
    colors = ['#3498db', '#2ecc71']
    x_pos = [-0.1,0.5]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(x_pos, values, color=colors, width=0.3, zorder=3)

    ax.set_title('Comparison: Deployed vs Executed Functions', weight='bold', fontsize=14, pad=25)
    ax.set_ylabel('Number of Functions', weight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontweight='bold')

    ax.set_xlim(-0.5, 0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    for i, v in enumerate(values):
        ax.text(x_pos[i], v + 1, str(v), ha='center', fontweight='bold')

    plt.savefig('../final_results_system.png')
    plt.show()


def analyze_and_visualize_comparison():
    try:
        df = pd.read_csv("../results.csv")
    except FileNotFoundError:
        return "Error: File not found."

    exp0 = df[df['experiment_number'] == 0].iloc[0]
    exp2 = df[df['experiment_number'] == 2].iloc[0]

    labels = ['Deployed', 'Correctly Executed']
    sub_optimal_vals = [exp0['num_deployed'], exp0['num_correctly_executed']]
    optimal_vals = [exp2['num_deployed'], exp2['num_correctly_executed']]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 7))

    rects1 = ax.bar(x - width / 2, sub_optimal_vals, width, label='Sub-optimal configuration', color='#3498db',
                    zorder=3)
    rects2 = ax.bar(x + width / 2, optimal_vals, width, label='Optimal configuration', color='#2ecc71', zorder=3)

    ax.set_title('Comparison: Sub-optimal vs Optimal Configuration', weight='bold', pad=45)
    ax.set_ylabel('Number of Functions', weight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, prop={'weight': 'bold'})

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('../comparison_configs.png')
    plt.show()

def create_full_sankey():
    try:
        df = pd.read_csv("../results_sankey.csv")
    except FileNotFoundError:
        print("Error: File '../results_sankey.csv' not found.")
        return

    row = df.iloc[0]

    labels = [
        "Deployed", "Executed", "Not Executed", "Final Function Correct",
        "Final Function Not Correct", "Final Function Correct", "Final Function Not Correct",
        "Debugged", "Not Debugged", "Debugged", "Not Debugged",
        "Debugged", "Not Debugged", "Debugged", "Not Debugged",
        "Original Function Correct", "Original Function Not Correct", "Original Function Correct", "Original Function Not Correct",
        "Original Function Correct", "Original Function Not Correct", "Original Function Correct", "Original Function Not Correct",
        "Original Function Correct", "Original Function Not Correct", "Original Function Correct", "Original Function Not Correct",
        "Original Function Correct", "Original Function Not Correct", "Original Function Correct", "Original Function Not Correct"
    ]

    links_data = [
        (0, 1, row['correctly_executed']),
        (0, 2, row['deployed_not_executed']),
        (1, 3, row['executed_final_correct']),
        (1, 4, row['executed_final_not_correct']),
        (2, 5, row['not_executed_final_correct']),
        (2, 6, row['not_executed_final_not_correct']),
        (3, 7, row['executed_final_correct_debugged']),
        (3, 8, row['executed_final_correct_not_debugged']),
        (4, 9, row['executed_final_not_correct_debugged']),
        (4, 10, row['executed_final_not_correct_not_debugged']),
        (5, 11, row['not_executed_final_correct_debugged']),
        (5, 12, row['not_executed_final_correct_not_debugged']),
        (6, 13, row['not_executed_final_not_correct_debugged']),
        (6, 14, row['not_executed_final_not_correct_not_debugged']),
        (7, 15, row['executed_final_correct_debugged_original_correct']),
        (7, 16, row['executed_final_correct_debugged_original_not_correct']),
        (8, 17, row['executed_final_correct_not_debugged_original_correct']),
        (8, 18, row['executed_final_correct_not_debugged_original_not_correct']),
        (9, 19, row['executed_final_not_correct_debugged_original_correct']),
        (9, 20, row['executed_final_not_correct_debugged_original_not_correct']),
        (10, 21, row['executed_final_not_correct_not_debugged_original_correct']),
        (10, 22, row['executed_final_not_correct_not_debugged_original_not_correct']),
        (11, 23, row['not_executed_final_correct_debugged_original_correct']),
        (11, 24, row['not_executed_final_correct_debugged_original_not_correct']),
        (12, 25, row['not_executed_final_correct_not_debugged_original_correct']),
        (12, 26, row['not_executed_final_correct_not_debugged_original_not_correct']),
        (13, 27, row['not_executed_final_not_correct_debugged_original_correct']),
        (13, 28, row['not_executed_final_not_correct_debugged_original_not_correct']),
        (14, 29, row['not_executed_final_not_correct_not_debugged_original_correct']),
        (14, 30, row['not_executed_final_not_correct_not_debugged_original_not_correct']),
    ]

    sources, targets, values = zip(*links_data)

    node_totals = [0] * len(labels)
    for src, tgt, val in links_data:
        if src == 0:
            node_totals[0] += val
        node_totals[tgt] += val

    updated_labels = [f"<b>{l} ({int(node_totals[i])})</b>" for i, l in enumerate(labels)]

    node_colors = ["grey"] * 1 + ["blue", "red"] * 1 + ["green", "orange"] * 2 + ["lightblue", "magenta"] * 4 + ["darkgreen", "darkred"] * 8

    link_colors = []
    for s_idx in sources:
        r, g, b, _ = mcolors.to_rgba(node_colors[s_idx])
        link_colors.append(f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.4)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=updated_labels,
            color=node_colors
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    )])

    experiment_id = row['experiment']
    fig.update_layout(
        title_text=f"Sankey Flow Diagram - Experiment: {experiment_id}",
        font=dict(size=12, color="black"),
        width=2000, height=1000
    )

    fig.show()

    dir_name = f"experiment_{experiment_id}/"
    os.makedirs(dir_name, exist_ok=True)
    output_path = os.path.join(dir_name, 'correct_functions_sankey.png')
    fig.write_image(output_path, width=3000, height=1200, scale=2)
    print(f"Diagram saved in {output_path}")

#create_detailed_sankey_diagram(experiment)
#create_sankey_from_dataframe(experiment)
#analyze_and_visualize_results()
#analyze_and_visualize_comparison()
create_full_sankey()
