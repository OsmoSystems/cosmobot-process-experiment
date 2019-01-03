from unittest.mock import sentinel

import pytest
import numpy as np

from osmo_camera.raw import metadata
from osmo_camera.raw.metadata import ExifTags
from osmo_camera.tiff import save
from osmo_camera import file_structure
from . import main as module

test_exif_tags = ExifTags(
    capture_datetime=None,
    iso=None,
    exposure_time=1.2
)


@pytest.fixture
def mock_correct_images(mocker):
    mocker.patch.object(metadata, 'get_exif_tags').return_value = test_exif_tags


class TestCorrectImages:
    def test_correct_images(self, mocker, mock_correct_images):
        mock_save_rgb_images = mocker.patch.object(module, 'save_rgb_images_by_filepath_with_suffix')

        rgbs_by_filepath = {
            sentinel.rgb_image_path_1: np.array([
                [[1, 10, 100], [2, 20, 200]],
                [[3, 30, 300], [4, 40, 400]]
            ])
        }

        actual = module.correct_images(
            rgbs_by_filepath,
            ROI_definition_for_intensity_correction=sentinel.ROI_definition,
            save_dark_frame_corrected_images=True,
            save_flat_field_corrected_images=True,
            save_intensity_corrected_images=True
        )

        expected = {
            sentinel.rgb_image_path_1: np.array([
                [[0.93752051, 9.93752051, 99.93752051], [1.93752051, 19.93752051, 199.93752051]],
                [[2.93752051, 29.93752051, 299.93752051], [3.93752051, 39.93752051, 399.93752051]]
            ])
        }

        np.testing.assert_array_almost_equal(actual[sentinel.rgb_image_path_1], expected[sentinel.rgb_image_path_1])
        assert mock_save_rgb_images.call_count == 3

    def test_save_rgb_images_by_filepath_with_suffix(self, mocker):
        mock_save_as_tiff = mocker.patch.object(save, 'as_tiff')
        mock_append_suffix_to_filepath_before_extension = mocker.patch.object(
            file_structure,
            'append_suffix_to_filepath_before_extension'
        )
        mock_replace_extension = mocker.patch.object(file_structure, 'replace_extension')

        rgbs_by_filepath = {
            sentinel.rgb_image_path_1: np.array([
                [[1, 10, 100], [2, 20, 200]],
                [[3, 30, 300], [4, 40, 400]]
            ]),
            sentinel.rgb_image_path_2: np.array([
                [[1, 10, 100], [2, 20, 200]],
                [[3, 30, 300], [4, 40, 400]]
            ])
        }

        module.save_rgb_images_by_filepath_with_suffix(rgbs_by_filepath, 'suffix')

        assert mock_save_as_tiff.call_count == 2
        assert mock_append_suffix_to_filepath_before_extension.call_count == 2
        assert mock_replace_extension.call_count == 2
