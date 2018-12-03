import numpy as np
import osmo_camera.correction.dark_frame as module


def test_correction_with_dark_frame():
    input_rgb = np.array([
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ])

    expected = [
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ]

    np.testing.assert_array_equal(module.apply_dark_frame_correction(input_rgb, input_rgb), expected)
