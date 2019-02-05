import numpy as np
import pytest

import osmo_camera.tiff.save as module


class TestGuardImageFitsIn32Bits():
    def setup_method(self):
        self.test_image = np.zeros(shape=(2, 2, 3))

    @pytest.mark.parametrize('name, in_range_value', [
        ('zero', 0),
        ('value within range', 1),
        ('value near min', -64),
        ('value near max', 63.9999),
    ])
    def test_does_not_raise_if_in_range(self, name, in_range_value):
        self.test_image[0][0][0] = in_range_value
        module._guard_rgb_image_fits_in_padded_range(self.test_image)

    @pytest.mark.parametrize('name, out_of_range_value', [
        ('value just below min', -64.1),
        ('value well below min', -10000),
        ('value just above max', 64),
        ('value well above max', 10000),
    ])
    def test_raises_if_out_of_range(self, name, out_of_range_value):
        self.test_image[0][0][0] = out_of_range_value
        with pytest.raises(module.DataTruncationError):
            module._guard_rgb_image_fits_in_padded_range(self.test_image)
