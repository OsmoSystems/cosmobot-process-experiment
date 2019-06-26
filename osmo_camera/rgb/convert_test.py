import numpy as np

import osmo_camera.rgb.convert as module


def test_convert_to_bgr():
    image = np.array(
        [
            [["r1", "g1", "b1"], ["r2", "g2", "b2"]],
            [["r3", "g3", "b3"], ["r4", "g4", "b4"]],
        ]
    )

    expected = np.array(
        [
            [["b1", "g1", "r1"], ["b2", "g2", "r2"]],
            [["b3", "g3", "r3"], ["b4", "g4", "r4"]],
        ]
    )

    np.testing.assert_array_equal(module.to_bgr(image), expected)


class TestConvertToPIL(object):
    def test_convert_to_PIL_no_warnings(self, mocker):
        mock_warning_logger = mocker.patch.object(module.logger, "warning")

        image = np.array([[[0, 0.5, 1], [1, 0.3, 0.1]]])

        expected = np.array([[[0, 127, 255], [255, 76, 25]]]).astype("uint8")

        PIL_image = module.to_PIL(image)
        np.testing.assert_array_equal(np.array(PIL_image), expected)
        mock_warning_logger.assert_not_called()

    def test_convert_to_PIL_with_warnings(self, mocker):
        mock_warning_logger = mocker.patch.object(module.logger, "warning")

        image = np.array([[[0, 0.5, 1], [1, 3, -0.1]]])

        expected = np.array(
            [
                # Check overflow values truncate properly:
                # 3 from input array should be the maximum value
                # -0.1 from input array should be the minimum value
                [[0, 127, 255], [255, 255, 0]]
            ]
        ).astype("uint8")

        PIL_image = module.to_PIL(image)
        np.testing.assert_array_equal(np.array(PIL_image), expected)
        mock_warning_logger.assert_called_once()
