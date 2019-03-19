import os
import pandas as pd

def _read_temperature_log(filepath):
    return pd.read_csv(filepath)

def temperature_for_experiment(experiment_directory_path):
    temperature_filepath = os.path.join(experiment_directory_path, 'temperature.csv')
    return _read_temperature_log(temperature_filepath)
