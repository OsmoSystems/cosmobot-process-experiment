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
