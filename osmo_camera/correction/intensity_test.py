from unittest.mock import sentinel

import numpy as np

import osmo_camera.correction.intensity as module


def test_apply_intensity_correction():
    input_rgb = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    actual = module.apply_intensity_correction(
        input_rgb,
        ROI_definition_for_intensity_correction=sentinel.ROI_definition
    )

    np.testing.assert_array_equal(actual, input_rgb)
