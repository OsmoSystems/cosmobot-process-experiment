import numpy as np
import pytest

from . import msorm as module


class TestTrimDataToStdev:
    @pytest.mark.parametrize(
        "name, sample, trim_stdev, expected",
        [
            ("edge case - uniform data", [1, 1, 1, 1, 1], 5, [1, 1, 1, 1, 1]),
            # fmt: off
            (
                'aggressive trim',
                [1] * 10 + [5, -5] + [10, -10],
                0.5,
                [1] * 10
            ),
            # fmt: on
            (
                "less aggressive trim",
                [1] * 10 + [5, -5] + [10, -10],
                2,
                [1] * 10 + [5, -5],
            ),
            ("trim everything when median not present", [1, 2, 3, 4], 0, []),
            ("trim everything except the median", [0, 1, 2], 0, [1]),
        ],
    )
    def test_various_cases(self, name, sample, trim_stdev, expected):
        np.testing.assert_array_equal(
            module._trim_data_to_stdev(np.array(sample), trim_stdev), np.array(expected)
        )


class TestMsorm:
    def test_unaffected_by_outliers(self):
        sample = np.array([5] * 100 + [-50, 100])

        actual = module.msorm(sample)
        expected = 5

        np.testing.assert_equal(actual, expected)

    def test_trim_stdev_large_enough__includes_outliers(self):
        sample = np.array([5] * 100 + [-50, 100])

        actual = module.msorm(sample, trim_stdev=10)

        # Outliers in this sample are biased towards the high end;
        # msorm should be noticably bigger than it would be if we filtered more aggressively
        msorm_without_outliers = 5
        assert actual > msorm_without_outliers + 0.1

    @pytest.mark.parametrize(
        "name,shape",
        # fmt: off
        [
            ('1D', (100,)),
            ('2D', (2, 50)),
            ('lots of Ds', (1, 2, 3, 4, 5)),
        ]
        # fmt: on
    )
    def test_accepts_any_array_shape(self, name, shape):
        non_flat_sample = np.full(shape, fill_value=5)

        actual = module.msorm(non_flat_sample)
        expected = 5

        assert actual == expected

    def test_zero_dimension_array_warns(self):
        sample = np.array([])

        # Underlying numpy code warns with RuntimeWarning when np.mean is called on empty array
        with pytest.warns(RuntimeWarning, match="empty"):
            module.msorm(sample)


EXAMPLE_RGB_IMAGE = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])

EXAMPLE_IMAGE_2_CHANNELS = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])


class TestImageMsorm:
    def test_returns_per_channel_stat(self):
        actual = module.image_msorm(EXAMPLE_RGB_IMAGE)
        expected = np.array([1, 2, 3])

        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "name, image",
        # fmt: off
        [
            ("too few channels", EXAMPLE_IMAGE_2_CHANNELS),
            (
                'too many channels',
                np.array([
                    [[1, 2, 3, 4], [1, 2, 3, 4]],
                    [[1, 2, 3, 4], [1, 2, 3, 4]],
                ])
            ),
            ("not an image", np.array([1, 2, 3])),
        ],
        # fmt: on
    )
    def test_blows_up_with_non_image_input(self, name, image):
        with pytest.raises(ValueError):
            module.image_msorm(image)


class TestImageStackMsorm:
    def test_returns_per_channel_stat(self):
        rgb_image_stack = np.array([EXAMPLE_RGB_IMAGE, EXAMPLE_RGB_IMAGE])

        actual = module.image_stack_msorm(rgb_image_stack)
        expected = np.array([1, 2, 3])

        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "name, image",
        [
            (
                "wrong number of channels",
                np.array([EXAMPLE_IMAGE_2_CHANNELS, EXAMPLE_IMAGE_2_CHANNELS]),
            ),
            ("not a stack", EXAMPLE_RGB_IMAGE),
        ],
    )
    def test_blows_up_with_non_image_input(self, name, image):
        with pytest.raises(ValueError):
            module.image_stack_msorm(image)
