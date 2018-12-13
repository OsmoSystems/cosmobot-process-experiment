from osmo_camera import raw, rgb

#  Constants to apply when calculating dark signal from final recommendation (note: these are in DNR):
#  https://docs.google.com/document/d/1xIgZxrC1qYUnwEGWt8yXnvWluEj51jpyIqArMJlNhrs/edit#
EXPOSURE_SLOPE = 6.81240234375e-05
DARK_OFFSET = 0.0623977451171875


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


def apply_dark_frame_correction_to_rgb_images(original_rgb_by_filepath, save_corrected_images=False):
    dark_frame_corrected_rgb_by_filepath = {
        image_path: _apply_dark_frame_correction(
            image_rgb,
            raw.metadata.get_exif_tags(image_path).exposure_time
        )
        for image_path, image_rgb in original_rgb_by_filepath.items()
    }

    if save_corrected_images:
        rgb.save.save_rgb_images_by_filepath_with_suffix(
            dark_frame_corrected_rgb_by_filepath,
            "_dark_adj"
        )

    return dark_frame_corrected_rgb_by_filepath
