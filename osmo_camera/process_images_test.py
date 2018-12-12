from unittest.mock import sentinel

import numpy as np

from osmo_camera.raw.metadata import ExifTags
from . import process_images as module


test_exif_tags = ExifTags(
    capture_datetime=None,
    iso=None,
    exposure_time=1.2
)


def test_correct_images():
    original_rgb_by_filepath = {
        sentinel.rgb_image_path_1: np.array([
            [[1, 10, 100], [2, 20, 200]],
            [[3, 30, 300], [4, 40, 400]]
        ])
    }

    exif_tags_by_filepath = {
        sentinel.rgb_image_path_1: test_exif_tags
    }

    actual = module.correct_images(
        original_rgb_by_filepath,
        exif_tags_by_filepath,
        ROI_definition_for_intensity_correction=sentinel.ROI_definition
    )

    expected = {
        sentinel.rgb_image_path_1: np.array([
            [[0.93752051, 9.93752051, 99.93752051], [1.93752051, 19.93752051, 199.93752051]],
            [[2.93752051, 29.93752051, 299.93752051], [3.93752051, 39.93752051, 399.93752051]]
        ])
    }

    np.testing.assert_array_almost_equal(actual[sentinel.rgb_image_path_1], expected[sentinel.rgb_image_path_1])


def test_get_ROI_statistics():
    mock_ROI = np.array([
        [[1, 10, 100], [2, 20, 200]],
        [[3, 30, 300], [4, 40, 400]]
    ])

    actual = module._get_ROI_statistics(mock_ROI)

    expected = {
        'r_cv': 1.118033988749895 / 2.5,
        'g_cv': 11.180339887498949 / 25.0,
        'b_cv': 111.80339887498948 / 250.0,
        'r_mean': 2.5,
        'g_mean': 25.0,
        'b_mean': 250.0,
        'r_median': 2.5,
        'g_median': 25.0,
        'b_median': 250.0,
        'r_min': 1,
        'g_min': 10,
        'b_min': 100,
        'r_max': 4,
        'g_max': 40,
        'b_max': 400,
        'r_stdev': 1.118033988749895,
        'g_stdev': 11.180339887498949,
        'b_stdev': 111.80339887498948,
        'r_percentile_99': 3.9699999999999998,
        'g_percentile_99': 39.699999999999996,
        'b_percentile_99': 396.99999999999994,
        'r_percentile_95': 3.8499999999999996,
        'g_percentile_95': 38.5,
        'b_percentile_95': 385.0,
        'r_percentile_90': 3.7,
        'g_percentile_90': 37.0,
        'b_percentile_90': 370.0,
        'r_percentile_75': 3.25,
        'g_percentile_75': 32.5,
        'b_percentile_75': 325.0,
        'r_percentile_50': 2.5,
        'g_percentile_50': 25.0,
        'b_percentile_50': 250.0,
        'r_percentile_25': 1.75,
        'g_percentile_25': 17.5,
        'b_percentile_25': 175.0
    }

    assert actual == expected
