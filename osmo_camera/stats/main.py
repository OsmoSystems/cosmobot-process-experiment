from functools import partial
from typing import Callable, Dict

import numpy as np

from osmo_camera.stats.msorm import image_msorm


# Running numpy calculations against this axis aggregates over the image for each channel, as color channels are axis=2
IMAGE_AXES = (0, 1)

image_mean = partial(np.mean, axis=IMAGE_AXES)
image_median = partial(np.median, axis=IMAGE_AXES)
image_min = partial(np.amin, axis=IMAGE_AXES)
image_max = partial(np.amax, axis=IMAGE_AXES)
image_stdev = partial(np.std, axis=IMAGE_AXES)


def image_outlier_warning(image):
    return image_msorm(image) - np.mean(image, axis=IMAGE_AXES) > 0.001


def image_coefficient_of_variation(image):
    return image_stdev(image) / image_mean(image)


# Type annotation clears things up for Mypy
roi_statistic_calculators: Dict[str, Callable] = {
    "msorm": image_msorm,
    "mean": image_mean,
    "median": image_median,
    "outlier_warning": image_outlier_warning,
    "min": image_min,
    "max": image_max,
    "stdev": image_stdev,
    "cv": image_coefficient_of_variation,
    **{
        f"percentile_{percentile}": partial(
            np.percentile, q=percentile, axis=IMAGE_AXES
        )
        for percentile in [99, 95, 90, 75, 50, 25]
    },
}
