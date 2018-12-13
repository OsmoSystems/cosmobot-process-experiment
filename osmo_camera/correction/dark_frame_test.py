import numpy as np

import osmo_camera.correction.dark_frame as module

exposure_seconds = 1.2


def test_calculate_dark_signal():
    actual = module._calculate_dark_signal_in_dnr(exposure_seconds)
    expected = 0.0624794
    np.testing.assert_almost_equal(actual, expected)


def test_apply_dark_frame_correction():
    input_rgb = np.array([
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ])

    actual = module._apply_dark_frame_correction(input_rgb, exposure_seconds)
    expected = np.array([
        [10.93752051, 11.93752051, 12.93752051, 13.93752051],
        [14.93752051, 15.93752051, 16.93752051, 17.93752051],
        [18.93752051, 19.93752051, 20.93752051, 21.93752051]
    ])

    np.testing.assert_array_almost_equal(actual, expected)
