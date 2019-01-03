from unittest.mock import sentinel

import numpy as np
import pandas as pd

import osmo_camera.correction.flat_field as module

rgb_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

rgb_image_series = pd.Series({
    sentinel.rgb_image_1: rgb_image,
    sentinel.rgb_image_2: rgb_image
})


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
        actual = module.apply_flat_field_correction_to_rgb_images(rgb_image_series)
        pd.testing.assert_series_equal(rgb_image_series, actual)
