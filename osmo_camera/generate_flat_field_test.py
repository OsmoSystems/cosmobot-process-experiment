import pytest
from unittest.mock import sentinel

import numpy as np
import pandas as pd

import osmo_camera.generate_flat_field as module
import osmo_camera.correction.dark_frame as dark_frame

rgb_image = np.array([
    [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]],
    [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]
])

rgb_image_series = pd.Series({
    sentinel.rgb_image_1: rgb_image,
    sentinel.rgb_image_2: rgb_image
})


@pytest.fixture
def mock_apply_dark_frame_correction_to_rgb_images(mocker):
    mocker.patch.object(dark_frame, 'apply_dark_frame_correction_to_rgb_images').return_value = rgb_image_series


def test_generate_flat_field(mock_apply_dark_frame_correction_to_rgb_images):
    actual = module.from_rgb_images(rgb_image_series)

    # The average RGB value of rgb_image is 0.2.  Expected calculated values are 0.1 / 0.2 = 0.5, 0.3 / 0.2 = 1.5
    expected = np.array([
        [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
        [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
    ])

    np.testing.assert_array_almost_equal(expected, actual)
