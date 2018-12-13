from unittest.mock import sentinel

import numpy as np

import osmo_camera.correction.flat_field as module


def test_apply_flat_field_correction():
    input_rgb = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120]
    ])

    actual = module._apply_flat_field_correction(
        input_rgb,
        sentinel.dark_frame_rgb,
        sentinel.flat_field_rgb
    )

    np.testing.assert_array_equal(actual, input_rgb)
