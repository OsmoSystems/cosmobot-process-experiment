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
    mocker.patch.object(module.rgb.filter, 'select_channels').return_value = sentinel.channel_selected_image
    mocker.patch.object(module.rgb.annotate, 'draw_ROIs_on_image').return_value = sentinel.roi_image
    mocker.patch.object(module.rgb.annotate, 'draw_text_on_image').return_value = sentinel.annotated_image


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
        module.rgb.annotate.draw_ROIs_on_image.return_value = test_array  # type: ignore

        annotated_scaled_image = module._open_filter_annotate_and_scale_image(
            sentinel.mock_filepath,
            sentinel.ROI_definition,
            image_scale_factor=0.5,
            color_channels='r',
            show_timestamp=False
        )

        np.testing.assert_array_equal(annotated_scaled_image, np.zeros((5, 5, 3)))

    def test_throws_when_no_timestamp_in_filename(self, mocker, mock_image_helper_functions):
        mocker.patch.object(module.rgb.convert, 'to_PIL')
        mocker.patch.object(module, 'scale_image')

        filepath = 'test_file_1'

        with pytest.raises(ValueError) as exception_info:
            module._open_filter_annotate_and_scale_image(
                filepath,
                sentinel.ROI_definition,
                image_scale_factor=1,
                color_channels='r',
                show_timestamp=True
            )

        assert f'\'{filepath}\' does not match format' in str(exception_info.value)

    def test_calls_draw_text_when_timestamp_in_filename(self, mocker, mock_image_helper_functions):
        filepath = '2019-01-01--00-00-00.jpeg'
        test_array = np.zeros((10, 10, 3))

        module.rgb.annotate.draw_text_on_image.return_value = test_array  # type: ignore

        module._open_filter_annotate_and_scale_image(
            filepath,
            sentinel.ROI_definition,
            image_scale_factor=1,
            color_channels='r',
            show_timestamp=True
        )

        module.rgb.annotate.draw_text_on_image.assert_called_with(  # type: ignore
            sentinel.roi_image,
            '2019-01-01 00:00:00'
        )

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
