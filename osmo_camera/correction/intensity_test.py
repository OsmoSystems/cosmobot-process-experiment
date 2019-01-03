from unittest.mock import sentinel

import numpy as np
import pandas as pd

import osmo_camera.correction.intensity as module

rgb_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])


class TestApplyIntensityCorrection:
    def test_apply_intensity_correction(self):
        input_rgb = rgb_image

        actual = module._apply_intensity_correction(
            input_rgb,
            ROI_definition_for_intensity_correction=sentinel.ROI_definition
        )

        np.testing.assert_array_equal(actual, input_rgb)

    def test_apply_intensity_correction_to_rgb_images(self):
        rgb_images = pd.Series({
            sentinel.rgb_image_1: rgb_image,
            sentinel.rgb_image_2: rgb_image
        })

        actual = module.apply_intensity_correction_to_rgb_images(
            rgb_images,
            sentinel.ROI_definition
        )

        pd.testing.assert_series_equal(rgb_images, actual)
