def _apply_flat_field_correction(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    # TODO (https://app.asana.com/0/819671808102776/926723356906177): implement
    return dark_frame_corrected_rgb


def apply_flat_field_correction_to_rgb_images(rgbs_by_filepath):
    return rgbs_by_filepath.apply(
        _apply_flat_field_correction,
        dark_frame_rgb=None,
        flat_field_rgb=None
    )
