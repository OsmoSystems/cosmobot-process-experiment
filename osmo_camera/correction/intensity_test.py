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
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]

    actual = module.apply_intensity_correction(input_rgb, correction_factor)
    np.testing.assert_array_almost_equal(actual, expected, decimal=1)
