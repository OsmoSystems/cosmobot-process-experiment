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


def _apply_flat_field_correction(dark_frame_corrected_rgb, flat_field_rgb):
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
    ''' Apply dark frame correction to an rgb image by subtracting a dark signal value

    Args:
        input_rgb: `RGB image` to correct
        exposure_seconds: number of seconds taken to expose image

    Returns:
        A rgb image that is dark frame corrected
    '''
    flat_field_rgb = open_flat_field_image(flat_field_filepath)

    return rgbs_by_filepath.apply(
        _apply_flat_field_correction,
        flat_field_rgb=flat_field_rgb
    )
