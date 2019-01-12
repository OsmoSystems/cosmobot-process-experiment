import warnings

import numpy as np
import pandas as pd
from scipy.stats import variation

from osmo_camera.correction.diagnostics import warn_if_any_true

# CV seems to be subject to rounding errors - make sure an insignificant change isn't flagged as an increase
MAX_CV_INCREASE = 0.000001

# Flat field probably shouldn't multiply the output by a huge amount or reduce it to a tiny smidgen
EXPECTED_MAX_FLAT_FIELD_FACTOR = 5
EXPECTED_MIN_FLAT_FIELD_FACTOR = 1 / EXPECTED_MAX_FLAT_FIELD_FACTOR


def get_flat_field_diagnostics(before, after, image_path):
    flat_field_difference = np.abs(after / before)
    diagnostics = pd.Series({
        # Mean coefficient of variation across all color channels
        'cv_before': variation(before, axis=None),
        'cv_after': variation(after, axis=None),
        'flat_field_factor_max': flat_field_difference.max(),
        'flat_field_factor_min': flat_field_difference.min(),
        'nan_values_after': np.count_nonzero(np.isnan(after)),
    })
    possible_warnings = pd.Series(
        {
            # Logically, the flat field should remove some real first-order effect
            # from the image so the Coefficient of Variation should decrease.
            'cv_increased': diagnostics['cv_after'] - diagnostics['cv_before'] > MAX_CV_INCREASE,
            'flat_field_factor_too_large': diagnostics['flat_field_factor_max'] > EXPECTED_MAX_FLAT_FIELD_FACTOR,
            'flat_field_factor_too_small': diagnostics['flat_field_factor_min'] < EXPECTED_MIN_FLAT_FIELD_FACTOR,
            'nan_values_present': diagnostics.nan_values_after,
        },
        # Force these values to be true/false - numbers in here are confusing & make warn_if_any_true mad
        dtype=np.bool_
    )

    warn_if_any_true(possible_warnings)
    return pd.concat([
        # Use dtype "object" to allow numbers as well as booleans to be in the result
        diagnostics.astype(np.object),
        possible_warnings
    ])


def _guard_flat_field_shape_matches(rgb_image, flat_field_rgb):
    if flat_field_rgb.shape != rgb_image.shape:
        raise ValueError(
            f'Flat field shape ({flat_field_rgb.shape}) does not match image shape ({rgb_image.shape})'
        )


def _apply_flat_field_correction(dark_frame_corrected_rgb, flat_field_rgb):
    _guard_flat_field_shape_matches(dark_frame_corrected_rgb, flat_field_rgb)
    return dark_frame_corrected_rgb * flat_field_rgb


def open_flat_field_image(flat_field_filepath):
    try:
        return np.load(flat_field_filepath)
    except OSError:  # Numpy raises an OSError when trying to open an invalid file type
        raise ValueError(
            f'Unable to load flat field image from path: {flat_field_filepath}.'
            f'Path must be the full path to a .npy file.'
        )


def apply_flat_field_correction_to_rgb_images(rgbs_by_filepath, flat_field_filepath):
    ''' Apply flat field correction to a Series of RGB images

    Args:
        rgbs_by_filepath: A pandas Series of `RGB image`s to correct
        flat_field_filepath: The full path of a .npy file to be used as the flat field image

    Returns:
        A Series of rgb images that have been flat-field corrected
    '''

    if flat_field_filepath is None:
        warnings.warn('No `flat_field_filepath` provided. Flat field correction *not* applied')
        return rgbs_by_filepath

    flat_field_rgb = open_flat_field_image(flat_field_filepath)

    return rgbs_by_filepath.apply(
        _apply_flat_field_correction,
        flat_field_rgb=flat_field_rgb
    )
