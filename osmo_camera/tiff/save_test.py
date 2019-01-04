import numpy as np
import pytest

import osmo_camera.tiff.save as module


class GuardImageFitsIn32Bits():
    def test_does_not_raise_if_in_range(self):
        image = np.array([
            [[-4, -4, -4], [-1, -1, -1]],
            [[1, 1, 1], [3.999999, 3.999999, 3.999999]]
        ])
        module._guard_image_fits_in_32_bits(image)

    def test_raises_if_out_of_range(self):
        image = np.array([
            [[-4.01, -4.01, -4.01], [-1, -1, -1]],
            [[1, 1, 1], [4.01, 4.01, 4.01]]
        ])
        with pytest.raises(ValueError):
            module._guard_image_fits_in_32_bits(image)
