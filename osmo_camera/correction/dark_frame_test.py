from unittest.mock import sentinel

import numpy as np

import osmo_camera.correction.dark_frame as module


def test_apply_dark_frame_correction():
    input_rgb = np.array([
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ])

    actual = module.apply_dark_frame_correction(input_rgb, sentinel.dark_frame_rgb)

    np.testing.assert_array_equal(actual, input_rgb)
