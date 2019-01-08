from unittest.mock import sentinel

import numpy as np
import pandas as pd
import pytest

import osmo_camera.correction.flat_field as module


def _replace_image_first_value(image, first_value):
    image_copy = np.copy(image)
    image_copy[0][0][0] = first_value
    return image_copy


class TestFlatFieldDiagnostics:
    mock_before_image = np.array([[[1, 1, 0.1], [3, 4, 5]]])

    @pytest.mark.parametrize('expected_warnings_raised, after_image', [
        (
            [],  # All clear
            mock_before_image * 2
        ),
        (
            ['cv_increased', 'flat_field_factor_too_large'],
            _replace_image_first_value(mock_before_image, 1000)
        ),
        (
            ['flat_field_factor_too_large'],
            mock_before_image * 100
        ),
        (
            ['flat_field_factor_too_small'],
            mock_before_image * 0.001
        ),
        (
            ['nan_values_present'],
            np.ones(mock_before_image.shape) * np.nan
        )
    ])
    def test_warning_cases(self, expected_warnings_raised, after_image, mocker):
        mock_warn_if_any_true = mocker.patch.object(module, 'warn_if_any_true')

        before = self.mock_before_image
        after = after_image

        # Numpy gets touchy when we throw around NaN values and such.
        # Quiet it down using this context manager:
        with np.errstate(invalid='ignore'):
            module.get_flat_field_diagnostics(before, after, mocker.sentinel.image_path)

        actual_warning_series = mock_warn_if_any_true.call_args[0][0]  # Indexing: First arg in first call

        expected_warning_series = pd.Series({
            'cv_increased': False,
            'flat_field_factor_too_large': False,
            'flat_field_factor_too_small': False,
            'nan_values_present': False,
        })
        for warning_name in expected_warnings_raised:
            expected_warning_series[warning_name] = True

        pd.testing.assert_series_equal(
            expected_warning_series, actual_warning_series
        )

    def test_returns_reasonable_values(self):
        actual = module.get_flat_field_diagnostics(
            self.mock_before_image,
            self.mock_before_image * 2,
            sentinel.image_path
        )

        actual_coefficient_of_variation = 0.7547445621912326
        expected = pd.Series({
            'cv_before': actual_coefficient_of_variation,
            'cv_after': actual_coefficient_of_variation,
            'flat_field_factor_max': 2,
            'flat_field_factor_min': 2,
            'nan_values_after': 0,
            'cv_increased': False,
            'flat_field_factor_too_large': False,
            'flat_field_factor_too_small': False,
            'nan_values_present': False,
        })

        pd.testing.assert_series_equal(actual, expected)


class TestFlatFieldCorrection:
    rgb_image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    def test_apply_flat_field_correction(self):
        input_rgb = self.rgb_image

        actual = module._apply_flat_field_correction(
            input_rgb,
            sentinel.dark_frame_rgb,
            sentinel.flat_field_rgb
        )

        np.testing.assert_array_equal(actual, input_rgb)

    def test_apply_intensity_correction_to_rgb_images(self):
        rgb_image_series = pd.Series({
            sentinel.rgb_image_1: self.rgb_image,
            sentinel.rgb_image_2: self.rgb_image
        })

        actual = module.apply_flat_field_correction_to_rgb_images(rgb_image_series)
        pd.testing.assert_series_equal(rgb_image_series, actual)
