import pytest
from unittest.mock import sentinel

import numpy as np
import pandas as pd

import osmo_camera.correction.flat_field as module
import osmo_camera.correction.dark_frame as dark_frame

rgb_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

rgb_images_pd_series = pd.Series({
    sentinel.rgb_image_1: rgb_image,
    sentinel.rgb_image_2: rgb_image
})


@pytest.fixture
def mock_apply_dark_frame_correction_to_rgb_images(mocker):
    mocker.patch.object(dark_frame, 'apply_dark_frame_correction_to_rgb_images').return_value = rgb_images_pd_series


def test_generate_flat_field(mock_apply_dark_frame_correction_to_rgb_images):
    actual = module.generate_flat_field(rgb_images_pd_series)
    expected = np.array([
        [6.5, 3.25, 2.166667, 1.625],
        [1.3, 1.083333, 0.928571, 0.8125],
        [0.722222, 0.65, 0.590909, 0.541667]
    ])

    np.testing.assert_array_almost_equal(expected, actual)


class TestFlatFieldCorrection:

    def test_apply_flat_field_correction(self):
        input_rgb = rgb_image

        actual = module._apply_flat_field_correction(
            input_rgb,
            sentinel.dark_frame_rgb,
            sentinel.flat_field_rgb
        )

        np.testing.assert_array_equal(actual, input_rgb)

    def test_apply_intensity_correction_to_rgb_images(self):
        rgb_images = rgb_images_pd_series
        actual = module.apply_flat_field_correction_to_rgb_images(rgb_images)
        pd.testing.assert_series_equal(rgb_images, actual)
