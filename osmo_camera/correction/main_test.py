from unittest.mock import sentinel

import numpy as np
import pytest

from osmo_camera.raw import metadata
from osmo_camera.raw.metadata import ExifTags
from osmo_camera.tiff import save
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


class TestCorrectImage:
    def test_correct_image(self, mocker, mock_exif_tags, mock_open_flat_field_image):
        mock_save_rgb_images = mocker.patch.object(module, 'save_rgb_image_with_suffix')

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

        actual_corrected_image, actual_diagnostics = module.correct_image(
            original_rgb_image=rgb_image,
            original_image_filepath=sentinel.rgb_image_path_1,
            flat_field_filepath_or_none=sentinel.flat_field_filepath,
            save_dark_frame_corrected_image=True,
            save_flat_field_corrected_image=True,
        )

        expected_corrected_image = np.array([
            [[0.412561, 0.412561, 0.412561], [0.213768, 0.237520, 0.475041], [0.412561, 0.412561, -0.187438]],
            [[0.337520, 0.875041, 0.483768], [0.50251, 0.56251, 0.50251], [0.483768, 0.875041, 0.337520]],
            [[0.412561, 0.412561, 0.412561], [1.275041, 0.573768, 0.637520], [0.412561, 0.412561, -0.187438]],
        ])

        np.testing.assert_allclose(actual_corrected_image, expected_corrected_image, atol=1e-5)
        mock_save_rgb_images.assert_called()
        # Spot check prefixing and presence of diagnostics
        assert not actual_diagnostics['dark_frame_min_value_increased']

    def test_save_rgb_image_with_suffix(self, mocker):
        mock_save_as_tiff = mocker.patch.object(save, 'as_tiff')

        rgb_image = np.array([
            [[1, 10, 100], [2, 20, 200]],
            [[3, 30, 300], [4, 40, 400]]
        ])

        module.save_rgb_image_with_suffix(
            rgb_image,
            'original image path.jpeg',
            'suffix'
        )

        mock_save_as_tiff.assert_called_with(rgb_image, 'original image pathsuffix.tiff')
