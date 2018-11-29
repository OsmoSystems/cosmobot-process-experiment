def _can_correct_with_flat_field(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    return dark_frame_corrected_rgb.shape() == dark_frame_rgb.shape() == flat_field_rgb.shape()

def flat_field_correction(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    if _can_correct_with_flat_field(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
        flat_field_corrected_rgb = dark_frame_corrected_rgb / (flat_field_rgb - dark_frame_rgb)
    else:
        raise ValueError("Not able to perform flat field correction")

    return flat_field_corrected_rgb
