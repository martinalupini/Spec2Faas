import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from app.Utils import *

config = get_config_data_full("../../config_test.yaml")
experiment = config["experiment_number"]

"""
def make_plot():
    df_csv = pd.read_csv('../results_old.csv')

    models = df_csv['model'].tolist()
    metrics = df_csv.columns.drop(['model']).tolist()

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

        df_value = df_csv.copy()
        models_plot = models
        valori = df_value[metric]

        plot_colors = [color_map[model] for model in models_plot]

        bars = ax.bar(models_plot, valori, color=plot_colors)

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
    """

def sankey_flow_diagram():
    df_csv = pd.read_csv('../results.csv')
    # Selecting only the current experiment
    df_csv = df_csv[df_csv['experiment_number'] == experiment].copy()

    for row in df_csv.itertuples(index=False):

        num_generated = row.num_generated
        num_deployed = row.num_deployed
        num_debugged = row.num_debugged
        num_correctly_executed = row.num_correctly_executed
        dir = "experiment_" + str(row.experiment_number) + "/"

        debugging_loss = num_generated - num_debugged
        deployment_loss = num_debugged - num_deployed
        execution_loss = num_deployed - num_correctly_executed

        # Define the Sankey diagram nodes and links
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Generated", "Debugged", "Deployed", "Correctly Executed", "Failed Debug", "Failed Deployment",
                       "Failed Execution"],
                color=["blue", "blue", "blue", "green", "red", "red", "red"]
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2],  # Indices of the source nodes
                target=[1, 4, 2, 5, 3, 6],  # Indices of the target nodes
                value=[num_debugged, debugging_loss, num_deployed, deployment_loss, num_correctly_executed, execution_loss]
                # Values of the flows
            ))])

        fig.update_layout(title_text="Sankey Diagram of the Execution Flow", font_size=12)
        plt.savefig( dir + 'comparison.png', dpi=300)
        fig.show()





def create_detailed_sankey_diagram(experiment):
    """
    Creates a detailed Sankey diagram for a specific experiment,
    including the flow values in the node labels.
    """
    try:
        # Assumes '../results.csv' is accessible relative to the script's location
        df_csv = pd.read_csv('../results.csv')
    except FileNotFoundError:
        print("Error File not found: Please ensure '../results.csv' exists.")
        return

    df_experiment = df_csv[df_csv['experiment_number'] == experiment].copy()

    if df_experiment.empty:
        print(f"No data for experiment {experiment}")
        return

    # Assuming the 'results.csv' has columns that match the flow values used:
    # 'num_debugged', 'num_generated_and_debugged', 'num_generated_debugged_deployed',
    # 'num_generated_debugged_not_deployed', 'num_generated_not_debugged_deployed',
    # 'num_generated_not_debugged_not_deployed', 'num_deployed', 'num_correctly_executed'
    # and total items is 164.

    for row in df_experiment.itertuples(index=False):
        dir_name = f"experiment_{row.experiment_number}/"
        os.makedirs(dir_name, exist_ok=True)

        execution_loss = row.num_deployed - row.num_correctly_executed
        total_items = 164 # Based on the link value 164 - row.num_debugged

        # Define the initial node labels and indices
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

        # --- Define Links and Values ---
        # The values here are critical as they define the flow *between* nodes.
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

        # --- Calculate Node Totals (Sum of incoming flows) ---
        node_totals = [0] * len(initial_labels)
        node_totals[0] = total_items # Total node has the fixed total value

        for src, tgt, val in zip(source_nodes, target_nodes, link_values):
            # For all nodes *except* the initial "Total" node (index 0),
            # the total value is the sum of its incoming links.
            if tgt != 0:
                 # Check if the flow value is valid
                if val >= 0:
                    node_totals[tgt] += val
                else:
                    # Handle cases where calculated values might be negative due to data inconsistencies
                    # or the original logic for the 3 -> 6 link (see NOTE above).
                    print(f"Warning: Negative flow value detected for link {initial_labels[src]} -> {initial_labels[tgt]}. Assuming 0 for calculation.")


        # --- Update Node Labels with Calculated Totals ---
        updated_labels = []
        for i, label in enumerate(initial_labels):
            updated_labels.append(f"{label} ({node_totals[i]})")


        # --- Rebuild Nodes and Links Dictionaries ---

        # Nodes (now with updated labels)
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

        # --- Create and Display Figure ---
        fig = go.Figure(data=[go.Sankey(node=nodes, link=links, arrangement='freeform')])

        fig.update_layout(
            font=dict(
                size=28,
                color="black",
                weight="bold"
            )
        )

        fig.show()

        # --- Save Figure ---
        try:
            output_path = os.path.join(dir_name, 'detailed_sankey_flow.png')
            fig.write_image(output_path, width=2000, height=800, scale=2)
            print(f"Diagram saved in {output_path}")
        except Exception as e:
            print(f"Errore in saving the diagram: {e}")


create_detailed_sankey_diagram(experiment)
