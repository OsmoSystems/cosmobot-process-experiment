import numpy as np

import osmo_camera.rgb.average as module


def test_can_average_ROI_spatially():
    input_rgb = np.array([
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[0, 0, 2], [0, 0, 2], [0, 0, 3]],
        [[0, 0, 3], [0, 0, 4], [0, 0, 5]],
    ])

    ROI_to_spatially_average = (1, 1, 2, 2)

    expected = 3.5
    blue_channel = 2

    assert module.spatial_average_of_roi(input_rgb, ROI_to_spatially_average)[blue_channel] == expected
