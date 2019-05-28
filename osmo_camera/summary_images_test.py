import os
from unittest.mock import sentinel

import numpy as np
import pytest
from PIL import Image

from . import summary_images as module

# Needed for `testdir` fixture
pytest_plugins = 'pytester'


@pytest.fixture
def mock_image_helper_functions(mocker):
    mocker.patch.object(module.raw.open, 'as_rgb').return_value = sentinel.rgb_image
    mocker.patch.object(module.rgb.filter, 'select_channels').return_value = sentinel.r_only_image
    mocker.patch.object(module, 'draw_ROIs_on_image').return_value = sentinel.roi_image
    mocker.patch.object(module, 'draw_text_on_image').return_value = sentinel.annotated_image


class TestImageFilepaths(object):
    def test_get_experiment_image_filepaths(self, mocker):
        mocker.patch.object(module, 'get_files_with_extension').return_value = [
            sentinel.image_file_1, sentinel.image_file_2
        ]

        image_paths = module.get_experiment_image_filepaths('./', ['path1', 'path2'])

        assert image_paths == [
            sentinel.image_file_1, sentinel.image_file_2,
            sentinel.image_file_1, sentinel.image_file_2
        ]


class TestScaleImage(object):
    test_image = Image.fromarray(np.zeros((10, 10, 3)).astype('uint8'))

    def test_scale_image_down(self):
        scaled_image = module.scale_image(self.test_image, 0.5)

        assert np.array(scaled_image).shape == (5, 5, 3)

    def test_scale_image_up(self):
        scaled_image = module.scale_image(self.test_image, 2)
        assert np.array(scaled_image).shape == (20, 20, 3)

    def test_scale_image_fractional(self):
        scaled_image = module.scale_image(self.test_image, 0.3)
        assert np.array(scaled_image).shape == (3, 3, 3)


class TestSummaryMediaGeneration(object):
    def test_open_filter_annotate_and_scale_image(self, mocker, mock_image_helper_functions):
        test_array = np.zeros((10, 10, 3))
        mocker.patch.object(module.os.path, 'basename').return_value = sentinel.mock_filename
        mocker.patch.object(module, 'filename_has_correct_datetime_format').return_value = True
        module.draw_ROIs_on_image.return_value = test_array  # type: ignore

        annotated_scaled_image = module._open_filter_annotate_and_scale_image(
            sentinel.mock_filepath,
            sentinel.ROI_definition,
            image_scale_factor=0.5,
            color_channels='r',
            show_timestamp=False
        )

        np.testing.assert_array_equal(annotated_scaled_image, np.zeros((5, 5, 3)))
        module.raw.open.as_rgb.assert_called_with(sentinel.mock_filepath)  # type: ignore
        module.rgb.filter.select_channels.assert_called_with(sentinel.rgb_image, 'r')  # type: ignore
        module.draw_ROIs_on_image.assert_called_with(sentinel.r_only_image, sentinel.ROI_definition)  # type: ignore
        module.draw_text_on_image.assert_not_called()  # type: ignore

    def test_draw_text_is_not_called(self, mocker, mock_image_helper_functions):
        mocker.patch.object(module.rgb.convert, 'to_PIL')
        mocker.patch.object(module, 'scale_image')

        module._open_filter_annotate_and_scale_image(
            'test_file_1',
            sentinel.ROI_definition,
            image_scale_factor=1,
            color_channels='r',
            show_timestamp=True
        )

        module.draw_text_on_image.assert_not_called()  # type: ignore

    def test_draw_text_is_called(self, mocker, mock_image_helper_functions):
        filepath = '2019-01-01--00-00-00.jpeg'
        test_array = np.zeros((10, 10, 3))

        module.draw_text_on_image.return_value = test_array  # type: ignore

        module._open_filter_annotate_and_scale_image(
            filepath,
            sentinel.ROI_definition,
            image_scale_factor=1,
            color_channels='r',
            show_timestamp=True
        )

        module.draw_text_on_image.assert_called_with(sentinel.roi_image, '2019-01-01 00:00:00')  # type: ignore

    def test_generate_summary_gif(self, testdir, mocker):
        mocker.patch.object(module, '_open_filter_annotate_and_scale_image').return_value = np.zeros((10, 10, 3))
        scale_factor = 4
        module.generate_summary_gif(
            ['test_file_1', 'test_file_2'],
            sentinel.ROI_definitions,
            name='test',
            image_scale_factor=scale_factor,
        )

        expected_filename = 'test.gif'

        assert os.path.isfile(expected_filename)

    def test_generate_summary_video(self, testdir, mocker):
        mocker.patch.object(module, '_open_filter_annotate_and_scale_image').return_value = np.zeros((10, 10, 3))
        scale_factor = 4
        module.generate_summary_video(
            ['test_file_1', 'test_file_2'],
            sentinel.ROI_definitions,
            name='test',
            image_scale_factor=scale_factor,
        )

        expected_filename = 'test.mp4'

        assert os.path.isfile(expected_filename)


class TestDrawTextOnImage(object):
    def test_output_has_content(self):
        test_image = np.zeros((5, 5, 3))
        test_text = '1'

        # These particular text + font settings should result in every pixel
        # int the image being covered with the text
        output_image = module.draw_text_on_image(
            test_image,
            test_text,
            text_color_rgb=(1, 1, 1),
            text_position=(-10, 10),
            font_scale=1
        )

        np.testing.assert_array_equal(output_image, np.ones((5, 5, 3)))
        # Ensure input is not mutated
        assert (test_image != output_image).any()
