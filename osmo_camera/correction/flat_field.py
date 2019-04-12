import warnings
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import variation

from osmo_camera.correction.diagnostics import warn_if_any_true

# CV seems to be subject to rounding errors - make sure an insignificant change isn't flagged as an increase
MAX_CV_INCREASE = 0.000001


def get_flat_field_diagnostics(before, after, image_path):
    ''' Produce diagnostics and raise warnings based on a flat-field-corrected image

    Documentation of individual diagnostics and warnings is in README.md in the project root.

    Args:
        before: RGB image before flat field correction
        after: RGB image after flat field correction
        image_path: path to original raw image. Used to look up EXIF data
    Returns:
        pandas Series of diagnostics and "red flag" warnings.
    Warns:
        uses the Warnings API and CorrectionWarning if any red flags are present.
    '''
    flat_field_difference = np.abs(after / before)
    diagnostics = pd.Series({
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


def apply_flat_field_correction(dark_frame_corrected_rgb, flat_field_rgb):
    ''' Apply flat field correction to an RGB image

    Args:
        dark_frame_corrected_rgb: A dark-frame-corrected RGB image
        flat_field_rgb: flat field RGB image

    Returns:
        rgb image that has been flat-field corrected
    '''
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


def load_flat_field_and_apply_correction(
        dark_frame_corrected_rgb_image,
        flat_field_filepath_or_none: Union[str, None],
):
    if flat_field_filepath_or_none is None:
        warnings.warn('No `flat_field_filepath` provided. Flat field correction *not* applied')
        return dark_frame_corrected_rgb_image

    flat_field_rgb = open_flat_field_image(flat_field_filepath_or_none)
    return apply_flat_field_correction(dark_frame_corrected_rgb_image, flat_field_rgb)
