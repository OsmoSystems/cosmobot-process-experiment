import numpy as np
import pandas as pd

from osmo_camera import raw
from osmo_camera.correction.diagnostics import warn_if_any_true

#  Constants to apply when calculating dark signal from final recommendation (note: these are in DNR):
#  https://docs.google.com/document/d/1xIgZxrC1qYUnwEGWt8yXnvWluEj51jpyIqArMJlNhrs/edit#
EXPOSURE_SLOPE = 6.81240234375e-05
DARK_OFFSET = 0.0623977451171875

# Training data used in the final recommendation
TRAINING_ISO = 100
TRAINING_EXPOSURE_MIN = 9e-6
TRAINING_EXPOSURE_MAX = 6

# Logically, a completely dark frame would have 50% of its pixels < 0 after dark frame subtraction
# Much more than that would be a big red flag
EXPECTED_MAX_NEGATIVE_PIXELS_AFTER_DARKFRAME = 0.51


def dark_frame_diagnostics(before, after, image_path):
    ''' Produce diagnostics and raise warnings based on a dark-frame corrected image

    Args:
        before: RGB image before dark frame correction
        after: RGB image after dark frame correction
        image_path: path to original raw image. Used to look up EXIF data
    Returns:
        pandas Series of diagnostics and "red flag" warnings.
    Warns:
        uses the Warnings API and CorrectionWarning if any red flags are present.
    '''
    exif = raw.metadata.get_exif_tags(image_path)
    exposure_time = exif.exposure_time
    iso = exif.iso

    diagnostics = pd.Series({
        'min_value_before': before.min(),
        'min_value_after': after.min(),
        'mean_value_before': before.mean(),
        'mean_value_after': after.mean(),
        'negative_pixels_pct_before': np.sum(before < 0) / before.size,
        'negative_pixels_pct_after': np.sum(after < 0) / after.size,
        'nan_values_after': np.count_nonzero(np.isnan(after)),
        'iso': iso,
        'exposure_time': exposure_time,
    })

    potential_warnings = pd.Series(
        {
            'exposure_out_of_training_range':
                not (TRAINING_EXPOSURE_MIN <= exposure_time <= TRAINING_EXPOSURE_MAX),
            'iso_mismatch_with_training': iso != TRAINING_ISO,
            'min_value_increased': diagnostics.min_value_after > diagnostics.min_value_before,
            'mean_value_increased': diagnostics.mean_value_after > diagnostics.mean_value_before,
            'too_many_negative_pixels_after':
                diagnostics.negative_pixels_pct_after > EXPECTED_MAX_NEGATIVE_PIXELS_AFTER_DARKFRAME,
            'nan_values_present': diagnostics.nan_values_after,
        },
        dtype=np.bool_
    )

    warn_if_any_true(potential_warnings)

    return pd.concat([
        diagnostics.astype(np.object), potential_warnings
    ])


def _calculate_dark_signal_in_dnr(exposure_seconds):
    ''' Calculate the dark signal introduced over the length of an exposure

    Args:
        exposure_seconds: number of seconds taken to expose image

    Returns:
        A value representing the dark signal that is normalized
    '''
    return ((EXPOSURE_SLOPE * exposure_seconds) + DARK_OFFSET)


def _apply_dark_frame_correction(input_rgb, exposure_seconds):
    ''' Apply dark frame correction to an rgb image by subtracting a dark signal value

    Args:
        input_rgb: `RGB image` to correct
        exposure_seconds: number of seconds taken to expose image

    Returns:
        A rgb image that is dark frame corrected
    '''
    dark_signal = _calculate_dark_signal_in_dnr(exposure_seconds)
    dark_frame_corrected_rgb = input_rgb - dark_signal
    return dark_frame_corrected_rgb


def apply_dark_frame_correction_to_rgb_images(rgbs_by_filepath):
    dark_frame_corrected_rgbs_by_filepath = pd.Series({
        image_path: _apply_dark_frame_correction(
            image_rgb,
            raw.metadata.get_exif_tags(image_path).exposure_time
        )
        for image_path, image_rgb in rgbs_by_filepath.items()
    })

    return dark_frame_corrected_rgbs_by_filepath
