import pkg_resources
from datetime import datetime
from unittest.mock import sentinel

import pandas as pd

from . import process_temperature_log as module


temperature_log_path = pkg_resources.resource_filename(
    "osmo_camera", "test_fixtures/temperature.csv"
)


class TestProcessTemperatureLog:
    def test_parses_raw_data_and_applies_calibration(self, mocker):
        mocker.patch("os.path.join").return_value = temperature_log_path
        mocker.patch.object(
            module, "temperature_given_digital_count_calibrated"
        ).return_value = sentinel.temperature

        actual = module.process_temperature_log(
            experiment_dir=sentinel.mock_experiment_dir,
            local_sync_directory_path=sentinel.mock_local_sync_path,
        )

        expected_temperature_data = pd.DataFrame(
            {
                "capture_timestamp": [
                    datetime(2019, 4, 30, 16, 29, 5),
                    datetime(2019, 4, 30, 16, 29, 11),
                ],
                "digital_count": [20007, 19993],
                "voltage": [2.5003263, 2.5025763],
                "temperature_c": [sentinel.temperature, sentinel.temperature],
            }
        )

        pd.testing.assert_frame_equal(
            actual, expected_temperature_data, check_less_precise=True
        )
