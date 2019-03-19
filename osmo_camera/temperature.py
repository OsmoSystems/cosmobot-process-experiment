import os
import pandas as pd

def _read_temperature_log(filepath):
    return pd.read_csv(filepath)

def temperature_for_experiment(local_sync_directory_path, experiment_name):
    experiment_directory_path = os.path.join(local_sync_directory_path, experiment_name)
    temperature_filepath = os.path.join(experiment_directory_path, 'temperature.csv')
    return _read_temperature_log(temperature_filepath)
