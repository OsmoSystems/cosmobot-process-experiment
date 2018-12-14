from unittest.mock import sentinel
import pytest
import numpy as np

import osmo_camera.correction.dark_frame as module

from osmo_camera.raw import metadata
from osmo_camera.raw.metadata import ExifTags

exposure_seconds = 1.2

test_exif_tags = ExifTags(
    capture_datetime=None,
    iso=None,
    exposure_time=exposure_seconds
)

rgb_image_1 = np.array([
    [11, 12, 13, 14],
    [15, 16, 17, 18],
    [19, 20, 21, 22]
])

rgb_image_2 = np.array([
    [12, 13, 14, 15],
    [16, 17, 18, 19],
    [20, 21, 22, 23]
])


@pytest.fixture
def mock_correct_images(mocker):
    mocker.patch.object(metadata, 'get_exif_tags').return_value = test_exif_tags


class TestDarkFrameCorrection:

    def test_calculate_dark_signal(self):
        actual = module._calculate_dark_signal_in_dnr(exposure_seconds)
        expected = 0.0624794
        np.testing.assert_almost_equal(actual, expected)

    def test_apply_dark_frame_correction(self):
        input_rgb = rgb_image_1

        actual = module._apply_dark_frame_correction(input_rgb, exposure_seconds)
        expected = np.array([
            [10.93752051, 11.93752051, 12.93752051, 13.93752051],
            [14.93752051, 15.93752051, 16.93752051, 17.93752051],
            [18.93752051, 19.93752051, 20.93752051, 21.93752051]
        ])

        np.testing.assert_array_almost_equal(actual, expected)

    def test_apply_dark_frame_correction_to_rgb_images(self, mock_correct_images):
        rgb_images = {
            sentinel.rgb_image_1: rgb_image_1,
            sentinel.rgb_image_2: rgb_image_2
        }

        actual = module.apply_dark_frame_correction_to_rgb_images(rgb_images)

        expected = {
            sentinel.rgb_image_1: np.array([
                [10.93752051, 11.93752051, 12.93752051, 13.93752051],
                [14.93752051, 15.93752051, 16.93752051, 17.93752051],
                [18.93752051, 19.93752051, 20.93752051, 21.93752051]
            ]),
            sentinel.rgb_image_2: np.array([
                [11.937521, 12.937521, 13.937521, 14.937521],
                [15.937521, 16.937521, 17.937521, 18.937521],
                [19.937521, 20.937521, 21.937521, 22.937521]
            ])
        }

        np.testing.assert_array_almost_equal(actual[sentinel.rgb_image_1], expected[sentinel.rgb_image_1])
        np.testing.assert_array_almost_equal(actual[sentinel.rgb_image_2], expected[sentinel.rgb_image_2])
