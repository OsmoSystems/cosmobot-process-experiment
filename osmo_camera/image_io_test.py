import pytest
import numpy as np

from osmo_camera import raw as raw_module
from osmo_camera import rgb as rgb_module


@pytest.fixture(scope='session')
def tmp_image_filepath(tmpdir_factory):
    return tmpdir_factory.mktemp("data").join("test_image.tiff")


def test_rgb_image_and_file_integration(tmp_image_filepath):
    actual_test_rgb_image = np.array([
        [[0.0, 0.0, 0.0], [0.499999, 0.499999, 0.499999]],
        [[0.699999, 0.699999, 0.699999], [0.999999, 0.999999, 0.999999]]
    ])

    # save rgb_image as tmp tiff
    tiff_tmp_filepath = 'test.tiff'
    rgb_module.save.as_uint16_tiff(actual_test_rgb_image, tiff_tmp_filepath)

    # read rgb_image from tmp tiff (should convert)
    expected_tmp_tiff_as_rgb_image = raw_module.open.as_uint16_tiff_as_rgb(tiff_tmp_filepath)
    assert np.testing.assert_array_almost_equal(actual_test_rgb_image, expected_tmp_tiff_as_rgb_image)
