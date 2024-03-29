import warnings
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

    @pytest.mark.parametrize(
        "expected_warnings_raised, after_image",
        [
            ([], mock_before_image * 2),  # All clear
            (["cv_increased"], _replace_image_first_value(mock_before_image, 1000)),
            (["nan_values_present"], np.ones(mock_before_image.shape) * np.nan),
        ],
    )
    def test_warning_cases(self, expected_warnings_raised, after_image, mocker):
        mock_warn_if_any_true = mocker.patch.object(module, "warn_if_any_true")

        before = self.mock_before_image
        after = after_image

        # Numpy gets touchy when we throw around NaN values and such.
        # Quiet it down using this context manager:
        with np.errstate(invalid="ignore"):
            module.get_flat_field_diagnostics(before, after, mocker.sentinel.image_path)

        # Indexing: First arg in first call
        actual_warning_series = mock_warn_if_any_true.call_args[0][0]

        expected_warning_series = pd.Series(
            {"cv_increased": False, "nan_values_present": False}
        )
        for warning_name in expected_warnings_raised:
            expected_warning_series[warning_name] = True

        pd.testing.assert_series_equal(expected_warning_series, actual_warning_series)

    def test_returns_reasonable_values(self):
        actual = module.get_flat_field_diagnostics(
            self.mock_before_image, self.mock_before_image * 2, sentinel.image_path
        )

        actual_coefficient_of_variation = 0.7547445621912326
        expected = pd.Series(
            {
                "cv_before": actual_coefficient_of_variation,
                "cv_after": actual_coefficient_of_variation,
                "flat_field_factor_max": 2,
                "flat_field_factor_min": 2,
                "nan_values_after": 0,
                "cv_increased": False,
                "nan_values_present": False,
            }
        )

        pd.testing.assert_series_equal(actual, expected)


class TestGuardFlatFieldShapeMatches:
    def test_raises_if_shape_does_not_match(self):
        with pytest.raises(ValueError):
            module._guard_flat_field_shape_matches(
                np.ones(shape=(1, 2, 3)), np.ones(shape=(4, 5, 6))
            )

    def test_does_not_raise_if_shape_matches(self):
        module._guard_flat_field_shape_matches(
            np.ones(shape=(1, 2, 3)), np.ones(shape=(1, 2, 3))
        )


class TestFlatFieldCorrection:
    # Approximate an actual vignetting effect
    rgb_image = np.array(
        [
            [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.2, 0.2, 0]],
            [[0.4, 0.5, 0.6], [0.9, 1, 0.9], [0.6, 0.5, 0.4]],
            [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7], [0.2, 0.2, 0]],
        ]
    )

    # Approximate an actual flat field image
    flat_field_rgb = np.array(
        [
            [[3, 3, 3], [0.9, 1, 2], [3, 3, 3]],
            [[1, 2, 0.9], [0.6, 0.6, 0.6], [0.9, 2, 1]],
            [[3, 3, 3], [2, 0.9, 1], [3, 3, 3]],
        ]
    )

    expected_flat_field_corrected_rgb = np.array(
        [
            [[0.6, 0.6, 0.6], [0.27, 0.3, 0.6], [0.6, 0.6, 0]],
            [[0.4, 1, 0.54], [0.54, 0.6, 0.54], [0.54, 1, 0.4]],
            [[0.6, 0.6, 0.6], [1.4, 0.63, 0.7], [0.6, 0.6, 0]],
        ]
    )

    def test_apply_flat_field_correction_with_identity_returns_original_image(self):
        input_rgb = self.rgb_image

        actual = module.apply_flat_field_correction(
            input_rgb, flat_field_rgb=np.ones(shape=input_rgb.shape)
        )

        np.testing.assert_array_equal(actual, input_rgb)

    def test_apply_flat_field_correction_multiplies(self):
        input_rgb = self.rgb_image

        actual = module.apply_flat_field_correction(input_rgb, self.flat_field_rgb)

        np.testing.assert_array_almost_equal(
            actual, self.expected_flat_field_corrected_rgb
        )

    def test_load_flat_field_and_apply_correction(self, mocker):
        mocker.patch.object(
            module, "open_flat_field_image"
        ).return_value = self.flat_field_rgb

        actual = module.load_flat_field_and_apply_correction(
            self.rgb_image, sentinel.flat_field_filepath
        )
        np.testing.assert_almost_equal(actual, self.expected_flat_field_corrected_rgb)

    def test_load_flat_field_and_apply_correction_raises_if_invalid_path(self):
        with pytest.raises(ValueError):
            module.load_flat_field_and_apply_correction(
                sentinel.rgb_image_series, flat_field_filepath_or_none="invalid.notnpy"
            )

    def test_load_flat_field_and_apply_correction_no_ops_and_warns_if_missing_path(
        self,
    ):
        with warnings.catch_warnings(record=True) as _warnings:
            actual = module.load_flat_field_and_apply_correction(
                sentinel.rgb_image_series, flat_field_filepath_or_none=None
            )

        # mypy thinks this could be None but it's not
        assert len(_warnings) == 1  # type: ignore
        assert actual == sentinel.rgb_image_series
