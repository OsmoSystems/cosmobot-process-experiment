import numpy as np

import image_basics as module


def test_crop_image():
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
    ])
    # start_col, start_row, cols, rows
    region = (1, 2, 2, 3)

    expected = [
        [10, 11],
        [14, 15],
        [18, 19],
    ]

    np.testing.assert_array_equal(module.crop_image(image, region), expected)
