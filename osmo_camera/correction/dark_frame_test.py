import numpy as np
import osmo_camera.correction.dark_frame as module


def test_correction_with_dark_frame():
    input_rgb = np.array([
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ])

    dark_frame_rgb = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    expected = [
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [10, 10, 10, 10]
    ]

    np.testing.assert_array_equal(module.dark_frame_correction(input_rgb, dark_frame_rgb), expected)
