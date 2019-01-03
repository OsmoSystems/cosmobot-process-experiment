import pytest
from unittest.mock import sentinel

import numpy as np
import pandas as pd

import osmo_camera.generate_flat_field as module
import osmo_camera.correction.dark_frame as dark_frame

rgb_image = np.array([
    [[0.072, 0.107, 0.235], [0.066, 0.104, 0.239]],
    [[0.069, 0.101, 0.227], [0.068, 0.100, 0.224]]
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
    expected = np.array([
        [[0.95486111, 0.96261682, 0.98404255], [1.04166667, 0.99038462, 0.96757322]],
        [[0.99637681, 1.01980198, 1.01872247], [1.01102941, 1.03, 1.03236607]]
    ])

    print(actual)
    print(expected)

    np.testing.assert_array_almost_equal(expected, actual)
