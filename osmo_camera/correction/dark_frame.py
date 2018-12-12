EXPOSURE_SLOPE = 0.069759
DARK_OFFSET = 63.895291
RAW_BIT_DEPTH = 2**10


def _calculate_dark_signal_in_dnr(exposure_seconds):
    ''' Calculate dark signal based on final recommendation:
        https://docs.google.com/document/d/1xIgZxrC1qYUnwEGWt8yXnvWluEj51jpyIqArMJlNhrs/edit#

    Args:
        exposure_seconds: number of seconds for exposure retrieved from exif_tags in image

    Returns:
        A value representing the dark signal that is normalized from DN (0-1024) to DNR (0-1)
    '''
    return ((EXPOSURE_SLOPE * exposure_seconds) + DARK_OFFSET) / RAW_BIT_DEPTH


def apply_dark_frame_correction(input_rgb, exposure_seconds):
    dark_signal = _calculate_dark_signal_in_dnr(exposure_seconds)
    dark_frame_corrected_rgb = input_rgb - dark_signal
    return dark_frame_corrected_rgb
