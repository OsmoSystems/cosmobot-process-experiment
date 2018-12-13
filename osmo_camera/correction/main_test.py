from unittest.mock import sentinel

import pytest
import numpy as np

from osmo_camera.raw import metadata
from osmo_camera.raw.metadata import ExifTags
from . import main as module

test_exif_tags = ExifTags(
    capture_datetime=None,
    iso=None,
    exposure_time=1.2
)


@pytest.fixture
def mock_get_exif_tags(mocker):
    mocker.patch.object(metadata, 'get_exif_tags').return_value = test_exif_tags


def test_correct_images(mock_get_exif_tags):
    original_rgb_by_filepath = {
        sentinel.rgb_image_path_1: np.array([
            [[1, 10, 100], [2, 20, 200]],
            [[3, 30, 300], [4, 40, 400]]
        ])
    }

    actual = module.correct_images(
        original_rgb_by_filepath,
        ROI_definition_for_intensity_correction=sentinel.ROI_definition,
        save_dark_frame_corrected_images=False,
        save_flat_field_corrected_images=False,
        save_intensity_corrected_images=False
    )

    expected = {
        sentinel.rgb_image_path_1: np.array([
            [[0.93752051, 9.93752051, 99.93752051], [1.93752051, 19.93752051, 199.93752051]],
            [[2.93752051, 29.93752051, 299.93752051], [3.93752051, 39.93752051, 399.93752051]]
        ])
    }

    np.testing.assert_array_almost_equal(actual[sentinel.rgb_image_path_1], expected[sentinel.rgb_image_path_1])
