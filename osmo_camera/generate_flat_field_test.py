import pytest
from unittest.mock import sentinel

import numpy as np
import pandas as pd

import osmo_camera.generate_flat_field as module
import osmo_camera.correction.dark_frame as dark_frame

RGB_IMAGE = np.array([
    [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]],
    [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]
])

RGB_IMAGE_SERIES = pd.Series({
    sentinel.rgb_image_1: RGB_IMAGE,
    sentinel.rgb_image_2: RGB_IMAGE
})


@pytest.fixture
def mock_apply_dark_frame_correction_to_rgb_images(mocker):
    mocker.patch.object(dark_frame, 'get_metadata_and_apply_dark_frame_correction').return_value = RGB_IMAGE


def test_generate_flat_field(mock_apply_dark_frame_correction_to_rgb_images):
    actual = module.from_rgb_images(RGB_IMAGE_SERIES)

    # rgb_image average RGB = 0.2.  Expected calculated values are 0.2 / 0.1 = 2.0, 0.2 / 0.3 = 0.666667
    expected = np.array([
        [[2.0, 2.0, 2.0], [0.666667, 0.666667, 0.666667]],
        [[2.0, 2.0, 2.0], [0.666667, 0.666667, 0.666667]]
    ])

    np.testing.assert_array_almost_equal(expected, actual)
