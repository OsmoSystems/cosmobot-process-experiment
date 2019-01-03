from . import dark_frame


def _apply_flat_field_correction(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    # TODO (https://app.asana.com/0/819671808102776/926723356906177): implement
    return dark_frame_corrected_rgb


def apply_flat_field_correction_to_rgb_images(rgbs_by_filepath):
    return rgbs_by_filepath.apply(
        _apply_flat_field_correction,
        dark_frame_rgb=None,
        flat_field_rgb=None
    )


def generate_flat_field(rgbs_by_filepath):
    # Subtract dark frame (using exposure of image) for all images
    dark_frame_adjusted_rgb_images = dark_frame.apply_dark_frame_correction_to_rgb_images(rgbs_by_filepath)

    # Average (image) of all dark frame adjusted images
    flat_field_mean_of_image_stack = dark_frame_adjusted_rgb_images.mean(axis=0)

    # Average value of averaged image
    flat_field_mean = flat_field_mean_of_image_stack.mean(axis=(0, 1))

    return flat_field_mean / flat_field_mean_of_image_stack
