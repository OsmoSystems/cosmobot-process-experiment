import pytest  # noqa: F401 (imported but unused)
import numpy as np

from osmo_camera import tiff as module


@pytest.mark.parametrize("name, test_rgb_image", [
    ('0 to 1 DNR', np.array([
        [[0.0, 0.0, 0.0], [0.499999, 0.499999, 0.499999]],
        [[0.699999, 0.699999, 0.699999], [0.999999, 0.999999, 0.999999]]
    ])),
    ('-1 to +1 DNR', np.array([
        [[-0.999999, -0.999999, -0.999999], [-0.699999, -0.699999, -0.699999]],
        [[0.699999, 0.699999, 0.699999], [0.999999, 0.999999, 0.999999]]
    ])),
    ('-3 to +3 DNR', np.array([
        [[-2.999999, -2.999999, -2.999999], [-2.699999, -2.699999, -2.699999]],
        [[2.699999, 2.699999, 2.699999], [2.999999, 2.999999, 2.999999]]
    ])),
])
def test_rgb_image_saved_to_tiff_file_and_loaded_retains_data(tmp_path, name, test_rgb_image):
    absolute_tolerance = 1.862645e-9  # 1 / 2^29 (quantization error when saving to and reading from an rgb image)

    # tmp_path provides an object of type PosixPath, tifffile expects a file path as a string
    tmp_filepath = str(tmp_path / 'test_tiff.tiff')
    module.save.as_tiff(test_rgb_image, tmp_filepath)

    # read rgb_image from tmp tiff (should convert)
    expected_tmp_tiff_as_rgb_image = module.open.as_rgb(tmp_filepath)
    np.testing.assert_allclose(test_rgb_image, expected_tmp_tiff_as_rgb_image, atol=absolute_tolerance)


def test_tiff_save_raises_if_image_out_of_range(tmp_path):
    # tmp_path provides an object of type PosixPath, tifffile expects a file path as a string
    tmp_filepath = str(tmp_path / 'test_tiff.tiff')

    test_rgb_image = np.array([
        [[-4.01, -4.01, -4.01], [-1, -1, -1]],
        [[1, 1, 1], [4.01, 4.01, 4.01]]
    ])

    with pytest.raises(ValueError):
        module.save.as_tiff(test_rgb_image, tmp_filepath)
