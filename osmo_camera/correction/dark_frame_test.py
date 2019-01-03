import numpy as np
import pandas as pd
import pytest
from unittest.mock import sentinel

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


@pytest.fixture
def mock_correct_images(mocker):
    mocker.patch.object(metadata, 'get_exif_tags').return_value = test_exif_tags


class TestDarkFrameDiagnostics:
    # TODO: tests for each failure case
    pass


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

    def test_apply_dark_frame_correction_to_rgb_images(self, mocker):
        mock__apply_dark_frame_correction = mocker.patch.object(module, '_apply_dark_frame_correction')
        mock_get_exif_tags = mocker.patch.object(metadata, 'get_exif_tags')

        rgb_images = pd.Series({
            sentinel.rgb_image_1: rgb_image_1,
            sentinel.rgb_image_2: rgb_image_1
        })

        module.apply_dark_frame_correction_to_rgb_images(rgb_images)
        assert mock__apply_dark_frame_correction.call_count == 2
        assert mock_get_exif_tags.call_count == 2
