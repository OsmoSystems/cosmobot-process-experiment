import pytest  # noqa: F401 (imported but unused)
import numpy as np

from osmo_camera import raw as raw_module
from osmo_camera import rgb as rgb_module


def test_rgb_image_and_file_integration(tmp_path):
    actual_test_rgb_image = np.array([
        [[0.0, 0.0, 0.0], [0.499999, 0.499999, 0.499999]],
        [[0.699999, 0.699999, 0.699999], [0.999999, 0.999999, 0.999999]]
    ])

    absolute_tolerance = 2.32831e-10  # 1/2^32 (quantization error when saving to and reading from a 32 bit image)

    # tmp_path provides an object of type PosixPath, tifffile expects a file path as a string
    tmp_filepath = str(tmp_path / 'test.tiff')
    rgb_module.save.as_uint32_tiff(actual_test_rgb_image, tmp_filepath)

    # read rgb_image from tmp tiff (should convert)
    expected_tmp_tiff_as_rgb_image = raw_module.open.as_uint32_tiff_as_rgb(tmp_filepath)
    np.testing.assert_allclose(actual_test_rgb_image, expected_tmp_tiff_as_rgb_image, atol=absolute_tolerance)
