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
    mocker.patch.object(dark_frame, 'get_metadata_and_apply_dark_frame_correction').return_value = rgb_image


def test_generate_flat_field(mock_apply_dark_frame_correction_to_rgb_images):
    actual = module.from_rgb_images(rgb_image_series)

    # rgb_image average RGB = 0.2.  Expected calculated values are 0.2 / 0.1 = 2.0, 0.2 / 0.3 = 0.666667
    expected = np.array([
        [[2.0, 2.0, 2.0], [0.666667, 0.666667, 0.666667]],
        [[2.0, 2.0, 2.0], [0.666667, 0.666667, 0.666667]]
    ])

    np.testing.assert_array_almost_equal(expected, actual)
