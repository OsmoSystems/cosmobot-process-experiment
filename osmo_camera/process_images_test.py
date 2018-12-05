from unittest.mock import sentinel

import numpy as np

from . import process_images as module


def test_correct_images():
    original_rgb_by_filepath = {
        sentinel.raw_path_1: np.array([
            [[1, 10, 100], [2, 20, 200]],
            [[3, 30, 300], [4, 40, 400]]
        ]),
        sentinel.raw_path_2: np.array([
            [[1, 10, 100], [2, 20, 200]],
            [[3, 30, 300], [4, 40, 400]]
        ])
    }

    corrected_rgb_images = module.correct_images(
        original_rgb_by_filepath,
        ROI_definition_for_intensity_correction=sentinel.ROI_definition
    )

    assert corrected_rgb_images == original_rgb_by_filepath


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
