def intensity_correction(flat_field_corrected_rgb, correction_factor):
    intensity_corrected_rgb = flat_field_corrected_rgb / correction_factor
    return intensity_corrected_rgb
