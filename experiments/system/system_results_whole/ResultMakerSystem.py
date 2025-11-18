import numpy as np
import pandas as pd
import os
from app.Utils import *
from experiments.faas_deployer.Utils import *
import matplotlib.pyplot as plt
import math

config = get_config_data_full("../../config_test.yaml")
experiment = config["experiment_number"]

directory = "experiment_" + str(experiment) + "/"
file_path = directory + "results.parquet"
csv_file = directory + "final_results.csv"

try:
    df = pd.read_parquet(file_path)

    #print(df.head(10))

    #print(df.info())

except FileNotFoundError:
    print(f"Errore: File non trovato a questo percorso: {file_path}")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")