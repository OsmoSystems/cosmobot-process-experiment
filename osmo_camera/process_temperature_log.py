import os

import pandas as pd

from osmo_camera.calibration.temperature import (
    temperature_given_digital_count_calibrated,
)


def process_temperature_log(
    experiment_dir,
    local_sync_directory_path,
    temperature_log_filename="temperature.csv",
):
    temperature_log_filepath = os.path.join(
        local_sync_directory_path, experiment_dir, temperature_log_filename
    )
    temperature_data = pd.read_csv(
        temperature_log_filepath, parse_dates=["capture_timestamp"]
    )

    temperature_data["temperature_c"] = temperature_data["digital_count"].apply(
        temperature_given_digital_count_calibrated
    )

    return temperature_data
