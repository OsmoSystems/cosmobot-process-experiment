import pytest
import numpy as np

import osmo_camera.rgb.filter as module


class TestFilterColorChannels(object):
    test_image = np.array([
        [['r1', 'g1', 'b1'], ['r2', 'g2', 'b2']],
        [['r3', 'g3', 'b3'], ['r4', 'g4', 'b4']]
    ])

    def test_select_all_channels(self):
        np.testing.assert_array_equal(module.select_channels(self.test_image, 'rgb'), self.test_image)

    def test_select_one_channel(self):
        test_image_copy = np.copy(self.test_image)
        expected = np.array([
            [['r1', 0, 0], ['r2', 0, 0]],
            [['r3', 0, 0], ['r4', 0, 0]]
        ])

        filtered_image = module.select_channels(test_image_copy, 'r')
        np.testing.assert_array_equal(filtered_image, expected)
        # Ensure the input image is unchanged
        np.testing.assert_array_equal(test_image_copy, self.test_image)
        assert (test_image_copy != filtered_image).any()

    def test_select_invalid_channel(self):
        with pytest.raises(ValueError):
            module.select_channels(self.test_image, 'MR')

    def test_select_invalid_capitalization(self):
        with pytest.raises(ValueError):
            module.select_channels(self.test_image, 'R')
