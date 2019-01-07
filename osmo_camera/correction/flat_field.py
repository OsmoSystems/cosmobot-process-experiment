import pandas as pd
from scipy.stats import variation

from osmo_camera.correction.diagnostics import warn_if_any_true


def flat_field_diagnostics(before, after, image_path):
    # TODO: unit tests
    diagnostics = pd.Series({
        # Mean coefficient of variation across all color channels
        'cv_before': variation(before).mean(),
        'cv_after': variation(after).mean(),
    })
    possible_warnings = pd.Series({
        # Logically, the flat field should remove some real first-order effect
        # from the image so the Coefficient of Variation should decrease.
        'cv_increased': diagnostics['cv_after'] > diagnostics['cv_before']
    })

    warn_if_any_true(possible_warnings)
    return pd.concat([diagnostics, possible_warnings])


def _apply_flat_field_correction(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    # TODO (https://app.asana.com/0/819671808102776/926723356906177): implement
    return dark_frame_corrected_rgb


def apply_flat_field_correction_to_rgb_images(rgbs_by_filepath):
    return rgbs_by_filepath.apply(
        _apply_flat_field_correction,
        dark_frame_rgb=None,
        flat_field_rgb=None
    )
