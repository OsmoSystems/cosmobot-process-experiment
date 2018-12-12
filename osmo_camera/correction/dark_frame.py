from ..constants import RAW_BIT_DEPTH

#  Constants to apply when calculating dark signal from final recommendation:
#  https://docs.google.com/document/d/1xIgZxrC1qYUnwEGWt8yXnvWluEj51jpyIqArMJlNhrs/edit#
EXPOSURE_SLOPE = 0.069759
DARK_OFFSET = 63.895291


def _calculate_dark_signal_in_dnr(exposure_seconds):
    ''' Calculate the dark signal introduced over the length of an exposure

    Args:
        exposure_seconds: number of seconds taken to expose image

    Returns:
        A value representing the dark signal that is normalized
    '''
    return ((EXPOSURE_SLOPE * exposure_seconds) + DARK_OFFSET) / RAW_BIT_DEPTH


def apply_dark_frame_correction(input_rgb, exposure_seconds):
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
