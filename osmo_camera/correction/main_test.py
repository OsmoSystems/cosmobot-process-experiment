from unittest.mock import sentinel

import numpy as np
import pandas as pd
import pytest

from osmo_camera.raw import metadata
from osmo_camera.raw.metadata import ExifTags
from osmo_camera.tiff import save
from osmo_camera import file_structure
from . import flat_field
from . import main as module

test_exif_tags = ExifTags(
    capture_datetime=None,
    iso=100,
    exposure_time=1.2
)


@pytest.fixture
def mock_exif_tags(mocker):
    mocker.patch.object(metadata, 'get_exif_tags').return_value = test_exif_tags


@pytest.fixture
def mock_open_flat_field_image(mocker):
    return mocker.patch.object(flat_field, 'open_flat_field_image')


class TestCorrectImages:
    def test_correct_images(self, mocker, mock_exif_tags, mock_open_flat_field_image):
        mock_save_rgb_images = mocker.patch.object(module, 'save_rgb_images_by_filepath_with_suffix')

        # Approximate an actual flat field image
        mock_open_flat_field_image.return_value = np.array([
            [[3, 3, 3], [.9, 1, 2], [3, 3, 3]],
            [[1, 2, .9], [.6, .6, .6], [.9, 2, 1]],
            [[3, 3, 3], [2, .9, 1], [3, 3, 3]],
        ])

        # Approximate an actual vignetting effect
        rgb_image = np.array([
            [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.2, 0.2, 0]],
            [[0.4, 0.5, 0.6], [0.9, 1, 0.9], [0.6, 0.5, 0.4]],
            [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7], [0.2, 0.2, 0]],
        ])
        rgbs_by_filepath = pd.Series({
            sentinel.rgb_image_path_1: rgb_image
        })

        actual_corrected_images, actual_diagnostics = module.correct_images(
            rgbs_by_filepath,
            ROI_definition_for_intensity_correction=sentinel.ROI_definition,
            flat_field_filepath=sentinel.flat_field_filepath,
            save_dark_frame_corrected_images=True,
            save_flat_field_corrected_images=True,
            save_intensity_corrected_images=True
        )

        expected_corrected_images = pd.Series({
            sentinel.rgb_image_path_1: np.array([
                [[0.04584017, 0.04584017, 0.04584017],
                 [0.26391167, 0.23752051, 0.11876025],
                 [0.04584017, 0.04584017, -0.0208265]],
                [[0.33752051, 0.21876025, 0.59724501],
                 [1.39586751, 1.56253418, 1.39586751],
                 [0.59724501, 0.21876025, 0.33752051]],
                [[0.04584017, 0.04584017, 0.04584017],
                 [0.31876025, 0.70835612, 0.63752051],
                 [0.04584017,  0.04584017, -0.0208265]]
            ])
        })

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(actual_corrected_images)

        pd.testing.assert_series_equal(
            actual_corrected_images,
            expected_corrected_images
        )
        assert mock_save_rgb_images.call_count == 3
        # Spot check prefixing and presence of diagnostics
        assert not actual_diagnostics['dark_frame_min_value_increased'].values[0]

    def test_save_rgb_images_by_filepath_with_suffix(self, mocker):
        mock_save_as_tiff = mocker.patch.object(save, 'as_tiff')
        mock_append_suffix_to_filepath_before_extension = mocker.patch.object(
            file_structure,
            'append_suffix_to_filepath_before_extension'
        )
        mock_replace_extension = mocker.patch.object(file_structure, 'replace_extension')

        rgbs_by_filepath = pd.Series({
            sentinel.rgb_image_path_1: np.array([
                [[1, 10, 100], [2, 20, 200]],
                [[3, 30, 300], [4, 40, 400]]
            ]),
            sentinel.rgb_image_path_2: np.array([
                [[1, 10, 100], [2, 20, 200]],
                [[3, 30, 300], [4, 40, 400]]
            ])
        })

        module.save_rgb_images_by_filepath_with_suffix(rgbs_by_filepath, 'suffix')

        assert mock_save_as_tiff.call_count == 2
        assert mock_append_suffix_to_filepath_before_extension.call_count == 2
        assert mock_replace_extension.call_count == 2
