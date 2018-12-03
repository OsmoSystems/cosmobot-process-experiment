import numpy as np
import osmo_camera.correction.flat_field as module


def test_correction_with_flat_field():

        dark_frame_corrected_rgb = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120]
        ])

        expected = [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120]
        ]

        np.testing.assert_array_equal(
            module.apply_flat_field_correction(
                dark_frame_corrected_rgb,
                dark_frame_corrected_rgb,
                dark_frame_corrected_rgb
            ),
            expected
        )
