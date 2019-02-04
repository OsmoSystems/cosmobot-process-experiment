import pytest  # noqa: F401 (imported but unused)
import numpy as np

from osmo_camera import tiff as module


@pytest.mark.parametrize("name, test_rgb_image", [
    ('Within [0, 1) DNR', np.array([
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
        [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    ])),
    ('Within padded [-4, 4) DNR', np.array([
        [[-64, -64, -64], [-0.5, -0.5, -0.5]],
        [[0.5, 0.5, 0.5], [63.999999, 63.999999, 63.999999]]
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


@pytest.mark.parametrize("name, test_rgb_image", [
    ('Below min', np.array([
        [[0, 0, 0], [-0.5, -0.5, -0.5]],
        [[0.5, 0.5, 0.5], [-64.01, -64.01, -64.01]]
    ])),
    ('Above max', np.array([
        [[0, 0, 0], [-0.5, -0.5, -0.5]],
        [[0.5, 0.5, 0.5], [64.01, 64.01, 64.01]]
    ])),
])
def test_tiff_save_raises_if_image_out_of_range(tmp_path, name, test_rgb_image):
    # tmp_path provides an object of type PosixPath, tifffile expects a file path as a string
    tmp_filepath = str(tmp_path / 'test_tiff.tiff')

    with pytest.raises(module.save.DataTruncationError):
        module.save.as_tiff(test_rgb_image, tmp_filepath)
