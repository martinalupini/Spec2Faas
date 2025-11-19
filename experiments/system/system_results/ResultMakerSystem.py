import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from app.Utils import *
from experiments.faas_deployer.Utils import *
import matplotlib.pyplot as plt
import math

config = get_config_data_full("../../config_test.yaml")
experiment = config["experiment_number"]

directory = "experiment_" + str(experiment) + "/"
file_path = directory + "results.parquet"
csv_file = "../results.csv"


try:
    df = pd.read_parquet(file_path)

    #print(df.info())

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")



columns = [
        'task_id', 'generated', 'deployed', 'correctly_executed', 'debugged', 'original_function_correct',
        'final_function_correct', 'test_correct' , 'prompt', 'signature', 'original_function', 'final_function',
        'CC_original', 'CC_final', 'CC_canonical', 'CoG_final', 'CoG_generated', 'CoG_canonical', 'time_assistant',
        'token_assistant', 'time_entry_point', 'token_entry_point', 'time_coder', 'token_coder', 'time_designer',
        'token_designer', 'time_executor', 'token_executor','time_debugger', 'token_debugger', 'time_deployer',
        'token_deployer', 'tests', 'coverage', 'debugging_attempts', 'number_messages_exchanged', 'canonical_solution', 'deployed_function'
    ]


def write_csv():

    num_generated = df['generated'].sum()
    num_deployed = df['deployed'].sum()
    num_correctly_executed = df['correctly_executed'].sum()
    num_debugged = df['debugged'].sum()

    num_not_generated = 164 - num_generated
    num_generated_and_debugged = df[df['generated'] & df['debugged']].shape[0]
    num_generated_and_not_debugged = num_generated - num_generated_and_debugged


    num_generated_debugged_deployed = df[df['generated'] & df['debugged'] & df['deployed']].shape[0]
    num_generated_not_debugged_deployed = df[df['generated'] & ~df['debugged'] & df['deployed']].shape[0]
    num_generated_debugged_not_deployed = num_deployed - num_generated_debugged_deployed
    num_generated_not_debugged_not_deployed = num_deployed - num_generated_not_debugged_deployed

    print(f"\n#Functions Generated: {num_generated}")
    print(f"\n#Functions Debugged: {num_debugged}")
    print(f"\n#Functions Deployed: {num_deployed}")
    print(f"\n#Functions Correctly Executed {num_correctly_executed}")


    results_data = {
        'experiment_number': experiment,
        'num_generated': num_generated,
        'num_deployed': num_deployed,
        'num_correctly_executed': num_correctly_executed,
        'num_debugged': num_debugged,
        'num_not_generated': num_not_generated,
        'num_generated_and_debugged': num_generated_and_debugged,
        'num_generated_and_not_debugged': num_generated_and_not_debugged,
        'num_generated_debugged_deployed': num_generated_debugged_deployed,
        'num_generated_not_debugged_deployed': num_generated_not_debugged_deployed,
        'num_generated_not_debugged_not_deployed': num_generated_not_debugged_not_deployed,
        'num_generated_debugged_not_deployed': num_generated_debugged_not_deployed,
    }

    # Non so perchè in questo file serviva index altrimenti ho errore
    results_df = pd.DataFrame(results_data, index=[0])

    file_exists = os.path.exists(csv_file)

    try:
        # Usa mode='a' (append) per aggiungere dati alla fine del file.
        # L'intestazione viene scritta solo se il file non esiste (header=not file_exists).
        results_df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

        if file_exists:
            print(f"\nAppended a new row to '{csv_file}'")
        else:
            print(f"\nCreated a new file and saved results to '{csv_file}'")

    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")


"""
def make_plot():
    df_csv = pd.read_csv('../results.csv')

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

    try:
        df_csv = pd.read_csv('../results.csv')
    except FileNotFoundError:
        print("Error File not found")
        return

    df_experiment = df_csv[df_csv['experiment_number'] == experiment].copy()

    if df_experiment.empty:
        print(f"No data for experiment {experiment}")
        return

    for row in df_experiment.itertuples(index=False):
        dir_name = f"experiment_{row.experiment_number}/"
        os.makedirs(dir_name, exist_ok=True)

        execution_loss = row.num_deployed - row.num_correctly_executed

        # Nodes
        nodes = dict(
            pad=25,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[
                "Generated",  # 0
                "Debugged",  # 1
                "Not Debugged",  # 2
                "Deployed",  # 3
                "Not Deployed",  # 4
                "Correctly Executed",  # 5
                "Execution Failed"  # 6
                "Not Generated" # 7
            ],
            color=[
                "grey",  # Generated
                "blue",  # Debugged
                "orange",  # Not Debugged
                "lightblue",  # Deployed
                "red",  # Not Deployed
                "green",  # Correctly Executed
                "darkred"  # Execution Failed
            ]
        )

        # Links between nodes
        links = dict(
            source=[
                0,  # Generated -> Debugged
                0,  # Generated -> Not Debugged
                1,  # Debugged -> Deployed
                1,  # Debugged -> Not Deployed
                2,  # Not Debugged -> Deployed
                2,  # Not Debugged -> Not Deployed
                3,  # Deployed -> Correctly Executed
                3,  # Deployed -> Execution Failed
            ],
            target=[
                1,  # Debugged
                2,  # Not Debugged
                3,  # Deployed
                4,  # Not Deployed
                3,  # Deployed
                4,  # Not Deployed
                5,  # Correctly Executed
                6,  # Execution Failed
            ],
            value=[
                row.num_generated_and_debugged,
                row.num_generated_and_not_debugged,
                row.num_generated_debugged_deployed,
                row.num_generated_debugged_not_deployed,
                row.num_generated_not_debugged_deployed,
                row.num_generated_not_debugged_not_deployed,
                row.num_correctly_executed,
                execution_loss
            ]
        )

        fig = go.Figure(data=[go.Sankey(node=nodes, link=links)])

        fig.update_layout(
            title_text=f" (Experiment {experiment} Flow Diagram)",
            font_size=14
        )

        fig.show()

        try:
            output_path = os.path.join(dir_name, 'detailed_sankey_flow.png')
            fig.write_image(output_path, width=1200, height=800, scale=2)
            print(f"Diagram saved in {output_path}")
        except Exception as e:
            print(f"Errore in saving the diagram: {e}")


create_detailed_sankey_diagram(experiment)



#write_csv()
