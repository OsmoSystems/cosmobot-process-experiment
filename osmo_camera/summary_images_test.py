import os
from unittest.mock import sentinel

from PIL import Image
import numpy as np
from . import summary_images as module

# Needed for `testdir` fixture
pytest_plugins = 'pytester'


def test_get_experiment_image_filepaths(mocker):
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
    def test_open_annotate_and_scale_image(self, mocker):
        test_array = np.zeros((10, 10, 3))
        mock_as_rgb = mocker.patch.object(module.raw.open, 'as_rgb')
        mock_as_rgb.return_value = sentinel.rgb_image
        mock_draw_ROIs_on_image = mocker.patch.object(module, 'draw_ROIs_on_image')
        mock_draw_ROIs_on_image.return_value = test_array

        annotated_scaled_image = module._open_annotate_and_scale_image(
            sentinel.mock_filepath,
            sentinel.ROI_definition,
            image_scale_factor=0.5
        )

        np.testing.assert_array_equal(annotated_scaled_image, np.zeros((5, 5, 3)))
        mock_as_rgb.assert_called_with(sentinel.mock_filepath)
        mock_draw_ROIs_on_image.assert_called_with(sentinel.rgb_image, sentinel.ROI_definition)

    def test_generate_summary_gif(self, testdir, mocker):
        mocker.patch.object(module, '_open_annotate_and_scale_image').return_value = np.zeros((10, 10, 3))
        scale_factor = 4
        module.generate_summary_gif(
            ['test_file_1', 'test_file_2'],
            sentinel.ROI_definitions,
            name='test',
            image_scale_factor=scale_factor,
        )

        expected_filename = 'test.gif'

        assert os.path.isfile(expected_filename)

        testdir.finalize()

    def test_generate_summary_video(self, testdir, mocker):
        mocker.patch.object(module, '_open_annotate_and_scale_image').return_value = np.zeros((10, 10, 3))
        scale_factor = 4
        module.generate_summary_video(
            ['test_file_1', 'test_file_2'],
            sentinel.ROI_definitions,
            name='test',
            image_scale_factor=scale_factor,
        )

        expected_filename = 'test.mp4'

        assert os.path.isfile(expected_filename)

        testdir.finalize()
