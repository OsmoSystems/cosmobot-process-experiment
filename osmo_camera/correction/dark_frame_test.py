import numpy as np

import osmo_camera.correction.dark_frame as module

exposure_seconds = 1.2


def test_calculate_dark_signal():
    actual = module._calculate_dark_signal(exposure_seconds)
    expected = 63.9790018
    np.testing.assert_almost_equal(actual, expected)


def test_apply_dark_frame_correction():
    input_rgb = np.array([
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ])

    actual = module.apply_dark_frame_correction(input_rgb, exposure_seconds)

    expected = np.array([
        [-52.979002, -51.979002, -50.979002, -49.979002],
        [-48.979002, -47.979002, -46.979002, -45.979002],
        [-44.979002, -43.979002, -42.979002, -41.979002]
    ])

    np.testing.assert_array_almost_equal(actual, expected)
