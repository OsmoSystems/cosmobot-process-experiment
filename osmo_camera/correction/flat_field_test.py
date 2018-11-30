import numpy as np
import osmo_camera.correction.flat_field as module


def test_correction_with_flat_field():

        dark_frame_corrected_rgb = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120]
        ])

        dark_frame_rgb = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])

        flat_field_rgb = np.array([
            [3, 4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13, 14]
        ])

        expected = [
            [ 5, 10, 15, 20],
            [25, 30, 35, 40],
            [45, 50, 55, 60]
        ]

        np.testing.assert_array_equal(
            module.flat_field_correction(
                dark_frame_corrected_rgb,
                dark_frame_rgb,
                flat_field_rgb
            ),
            expected
        )