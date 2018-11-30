import numpy as np
import osmo_camera.correction.intensity as module

def test_intensity_correction():
    input_rgb = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    correction_factor = 1.1

    expected = [
        [1.1, 2.2, 3.3, 4.4],
        [5.5, 6.6, 7.7, 8.8],
        [9.9, 11.0, 12.1, 13.2]
    ]

    actual = module.intensity_correction(input_rgb, correction_factor)
    np.testing.assert_array_equal(actual, expected)