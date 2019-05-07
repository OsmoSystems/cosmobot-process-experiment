import numpy as np

import osmo_camera.rgb.convert as module


def test_convert_to_bgr():
    image = np.array([
        [['r1', 'g1', 'b1'], ['r2', 'g2', 'b2']],
        [['r3', 'g3', 'b3'], ['r4', 'g4', 'b4']]
    ])

    expected = np.array([
        [['b1', 'g1', 'r1'], ['b2', 'g2', 'r2']],
        [['b3', 'g3', 'r3'], ['b4', 'g4', 'r4']]
    ])

    np.testing.assert_array_equal(module.to_bgr(image), expected)


def test_convert_to_int():
    image = np.array([
        [[0, 0.5, 1], [1, 3, 0.1]]
    ])

    expected = np.array([
        # Check overflow values truncate properly:
        # 3 * 255 = 765 -> uses 255 as multiplier as this is max value of uint8
        # 765 % 256 = 253 -> uses 256 a modulus as this is the # of uint8s
        [[0, 127, 255], [255, 253, 25]]
    ]).astype('uint8')

    np.testing.assert_array_equal(module.to_int(image), expected)
