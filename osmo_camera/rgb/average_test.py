import numpy as np

import osmo_camera.rgb.average as module


def test_can_average_roi_spatially():
    input_rgb = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    roi_to_spatially_average = (1, 1, 2, 2)

    expected = 8.5
    assert module.spatial_average_of_roi(input_rgb, roi_to_spatially_average) == expected
