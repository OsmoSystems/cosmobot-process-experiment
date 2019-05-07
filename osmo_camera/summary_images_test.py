import os
import pytest
from unittest.mock import sentinel

import numpy as np
from . import summary_images as module

# Needed for `testdir` fixture
pytest_plugins = "pytester"


@pytest.fixture
def mock_image_array():
    return module.rgb.convert.to_int(np.random.rand(10, 10, 3))


@pytest.fixture
def mock_side_effects(mocker, mock_image_array):
    mocker.patch.object(module.raw.open, 'as_rgb').return_value = sentinel.rgb_image
    mocker.patch.object(module, 'draw_ROIs_on_image').return_value = sentinel.rgb_image_with_ROI_definitions
    mocker.patch.object(module.rgb.convert, 'to_int').return_value = sentinel.PIL_compatible_image
    mock_scale_image = mocker.patch.object(module, 'scale_image')
    mock_scale_image.return_value = mock_image_array
    return {'mock_scale_image': mock_scale_image}


def test_scale_image(mock_image_array):
    scaled_image = module.scale_image(mock_image_array, 2)

    assert scaled_image.shape == (5, 5, 3)


def test_gather_experiment_images(mocker):
    mocker.patch.object(module, 'get_files_with_extension').return_value = [
        sentinel.image_file_1, sentinel.image_file_2]

    image_paths = module.gather_experiment_images(['path1', 'path2'], './')

    assert sentinel.image_file_1 in image_paths
    assert sentinel.image_file_2 in image_paths
    assert len(image_paths) == 4


def test_generate_summary_gif(testdir, mock_side_effects):
    scale_factor = 4
    module.generate_summary_gif(
        ['test_file_1', 'test_file_2'],
        sentinel.ROI_definitions,
        name="test",
        image_scale_factor=scale_factor,
    )

    expected_filename = 'test.gif'

    assert os.path.isfile(expected_filename)
    mock_side_effects['mock_scale_image'].assert_called_with(sentinel.PIL_compatible_image, scale_factor)

    testdir.finalize()


def test_generate_summary_video(testdir, mock_side_effects):
    scale_factor = 4
    module.generate_summary_video(
        ['test_file_1', 'test_file_2'],
        sentinel.ROI_definitions,
        name="test",
        image_scale_factor=scale_factor,
    )

    expected_filename = 'test.mp4'

    assert os.path.isfile(expected_filename)
    mock_side_effects['mock_scale_image'].assert_called_with(sentinel.PIL_compatible_image, scale_factor)

    testdir.finalize()
