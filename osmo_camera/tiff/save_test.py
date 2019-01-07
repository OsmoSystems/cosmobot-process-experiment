import numpy as np
import pytest

from osmo_camera.constants import DNR_TO_TIFF_FACTOR
import osmo_camera.tiff.save as module


class TestGuardImageFitsIn32Bits():
    @pytest.mark.parametrize("name, in_range_value", [
        ("value within range", 1),
        ("value near min", -4.1),
        ("value near max", 3.9999),
    ])
    def test_does_not_raise_if_in_range(self, name, in_range_value):
        image = np.array([
            [[in_range_value * DNR_TO_TIFF_FACTOR, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ])
        module._guard_image_fits_in_32_bits(image)

    @pytest.mark.parametrize("name, out_of_range_value", [
        ("value just below min", -4.1),
        ("value well below min", -10000),
        ("value just above max", 4),
        ("value well above max", 10000),
    ])
    def test_raises_if_out_of_range(self, name, out_of_range_value):
        image = np.array([
            [[out_of_range_value * DNR_TO_TIFF_FACTOR, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ])
        with pytest.raises(module.DataTruncationError):
            module._guard_image_fits_in_32_bits(image)
