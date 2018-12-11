import numpy as np
import pytest

from . import msorm as module


class TestTrimDataToStdev:
    @pytest.mark.parametrize('name, sample, trim_stdev, expected', [
        ('edge case - uniform data', [1, 1, 1, 1, 1], 5, [1, 1, 1, 1, 1]),
        (
            'aggressive trim',
            [1] * 10 + [5, -5] + [10, -10],
            0.5,
            [1] * 10
        ),
        (
            'less aggressive trim',
            [1] * 10 + [5, -5] + [10, -10],
            2,
            [1] * 10 + [5, -5]
        ),
        ('trim everything', [1, 2, 3, 4], 0, []),
    ])
    def test_various_cases(self, name, sample, trim_stdev, expected):
        np.testing.assert_array_equal(
            module._trim_data_to_stdev(np.array(sample), trim_stdev),
            np.array(expected)
        )


class TestMsorm:
    def test_unaffected_by_outliers(self):
        sample = np.array([5] * 100 + [-50, 100])

        actual = module.msorm(sample)
        expected = 5

        np.testing.assert_almost_equal(actual, expected)

    def test_msorm_trim_stdev_large_enough__includes_outliers(self):
        sample = np.array([5] * 100 + [-50, 100])

        actual = module.msorm(sample, 10)

        # Outliers in this sample are biased towards the high end;
        # msorm should be noticably bigger than it would be if we filtered more aggressively
        msorm_without_outliers = 5
        assert actual > msorm_without_outliers + 0.1

    def test_msorm_on_non_flat_array__blows_up(self):
        sample = np.array([
            [1, 2, 3], [1, 2, 3]
        ])

        with pytest.raises(ValueError, match='flat'):
            module.msorm(sample)


rgb_image = np.array([
    [[1, 2, 3],  [1, 2, 3]],
    [[1, 2, 3],  [1, 2, 3]],
])

image_2_channels = np.array([
    [[1, 2],  [1, 2]],
    [[1, 2],  [1, 2]],
])


class TestImageMsorm:
    def test_returns_per_channel_stat(self):
        actual = module.image_msorm(rgb_image)
        expected = np.array([1, 2, 3])

        np.testing.assert_array_almost_equal(actual, expected)

    @pytest.mark.parametrize('name, image', [
        (
            'wrong number of channels',
            image_2_channels
        ),
        (
            'not an image',
            np.array([1, 2, 3])
        ),
    ])
    def test_blows_up_with_non_image_input(self, name, image):
        with pytest.raises(ValueError):
            module.image_msorm(image)


class TestImageStackMsorm:
    def test_returns_per_channel_stat(self):
        rgb_image = np.array([
            [[1, 2, 3],  [1, 2, 3]],
            [[1, 2, 3],  [1, 2, 3]],
        ])
        rgb_image_stack = np.array([
            rgb_image, rgb_image
        ])

        actual = module.image_stack_msorm(rgb_image_stack)
        expected = np.array([1, 2, 3])

        np.testing.assert_array_almost_equal(actual, expected)

    @pytest.mark.parametrize('name, image', [
        (
            'wrong number of channels',
            np.array([image_2_channels, image_2_channels])
        ),
        (
            'not a stack',
            rgb_image
        ),
    ])
    def test_blows_up_with_non_image_input(self, name, image):
        with pytest.raises(ValueError):
            module.image_stack_msorm(image)
