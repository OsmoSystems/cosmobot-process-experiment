def flat_field_correction(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    flat_field_corrected_rgb = dark_frame_corrected_rgb / (flat_field_rgb - dark_frame_rgb)
    return flat_field_corrected_rgb
