from unittest.mock import sentinel

import numpy as np
import pandas as pd
import pytest

import osmo_camera.correction.dark_frame as module
from osmo_camera.raw import metadata
from osmo_camera.raw.metadata import ExifTags


exposure_seconds = 1.2

test_exif_tags = ExifTags(
    capture_datetime=None, iso=100, exposure_time=exposure_seconds
)

# fmt: off
rgb_image_1 = np.array([
    [11, 12, 13, 14],
    [15, 16, 17, 18],
    [19, 20, 21, 22]
])
# fmt: on


@pytest.fixture
def mock_normal_exif_tags(mocker):
    mocker.patch.object(metadata, "get_exif_tags").return_value = test_exif_tags


class TestDarkFrameDiagnostics:
    mock_before_image = np.array([[[0, 1, 2], [3, 4, 5]]])

    @pytest.mark.parametrize(
        "expected_warnings_raised, after_image",
        [
            ([], mock_before_image - 0.00001),  # All clear
            (["min_value_increased", "mean_value_increased"], mock_before_image + 1),
            (["too_many_negative_pixels_after"], -mock_before_image),
            (["nan_values_present"], np.ones(mock_before_image.shape) * np.nan),
        ],
    )
    def test_non_exif_warning_cases(
        self, expected_warnings_raised, after_image, mocker, mock_normal_exif_tags
    ):
        mock_warn_if_any_true = mocker.patch.object(module, "warn_if_any_true")

        before = self.mock_before_image
        after = after_image

        # Numpy gets touchy when we throw around NaN values and such.
        # Quiet it down using this context manager:
        with np.errstate(invalid="ignore"):
            module.get_dark_frame_diagnostics(before, after, mocker.sentinel.image_path)

        # Indexing: First arg in first call
        actual_warning_series = mock_warn_if_any_true.call_args[0][0]

        expected_warning_series = pd.Series(
            {
                "exposure_out_of_training_range": False,
                "iso_mismatch_with_training": False,
                "min_value_increased": False,
                "mean_value_increased": False,
                "too_many_negative_pixels_after": False,
                "nan_values_present": False,
            }
        )
        for warning_name in expected_warnings_raised:
            expected_warning_series[warning_name] = True

        pd.testing.assert_series_equal(expected_warning_series, actual_warning_series)

    def test_warning_cases_from_exif(self, mocker):
        mock_warn_if_any_true = mocker.patch.object(module, "warn_if_any_true")

        mocker.patch.object(
            module.raw.metadata, "get_exif_tags"
        ).return_value = ExifTags(
            capture_datetime=None,
            iso=10000000000000,
            exposure_time=9999999999999999999999999999,
        )

        before = self.mock_before_image
        after = before

        module.get_dark_frame_diagnostics(before, after, mocker.sentinel.image_path)

        actual_warning_series = mock_warn_if_any_true.call_args[0][0]

        expected_warning_series = pd.Series(
            {
                "exposure_out_of_training_range": True,
                "iso_mismatch_with_training": True,
                "min_value_increased": False,
                "mean_value_increased": False,
                "too_many_negative_pixels_after": False,
                "nan_values_present": False,
            }
        )

        pd.testing.assert_series_equal(expected_warning_series, actual_warning_series)

    def test_returns_reasonable_values(self, mock_normal_exif_tags):
        actual = module.get_dark_frame_diagnostics(
            self.mock_before_image, self.mock_before_image - 0.001, sentinel.image_path
        )

        expected = pd.Series(
            {
                "min_value_before": 0,
                "min_value_after": -0.001,
                "mean_value_before": 2.5,
                "mean_value_after": 2.499,
                "negative_pixels_pct_before": 0,
                "negative_pixels_pct_after": 0.166667,
                "nan_values_after": 0,
                "iso": 100,
                "exposure_time": 1.2,
                "exposure_out_of_training_range": False,
                "iso_mismatch_with_training": False,
                "min_value_increased": False,
                "mean_value_increased": False,
                "too_many_negative_pixels_after": False,
                "nan_values_present": False,
            }
        )

        pd.testing.assert_series_equal(actual, expected)


class TestDarkFrameCorrection:
    def test_calculate_dark_signal(self):
        actual = module.calculate_dark_signal_in_dnr(exposure_seconds)
        expected = 0.0624794
        np.testing.assert_almost_equal(actual, expected)

    def test_apply_dark_frame_correction(self):
        input_rgb = rgb_image_1

        actual = module.apply_dark_frame_correction(input_rgb, exposure_seconds)
        expected = np.array(
            [
                [10.93752051, 11.93752051, 12.93752051, 13.93752051],
                [14.93752051, 15.93752051, 16.93752051, 17.93752051],
                [18.93752051, 19.93752051, 20.93752051, 21.93752051],
            ]
        )

        np.testing.assert_array_almost_equal(actual, expected)

    def test_get_metadata_and_apply_dark_frame_correction_calls_appropriate_functions(
        self, mocker
    ):
        mock_apply_dark_frame_correction = mocker.patch.object(
            module, "apply_dark_frame_correction"
        )
        mock_get_exif_tags = mocker.patch.object(metadata, "get_exif_tags")

        module.get_metadata_and_apply_dark_frame_correction(
            rgb_image_1, sentinel.filename
        )

        mock_apply_dark_frame_correction.assert_called()
        mock_get_exif_tags.assert_called()
