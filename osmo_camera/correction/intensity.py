import numpy

def _can_correct_for_intensity(flat_field_corrected_rgb, correction_factor):
    return numpy.array(flat_field_corrected_rgb.ndim) == 3 and correction_factor > 0

def intensity_correction(flat_field_corrected_rgb, correction_factor):
    if _can_correct_for_intensity(flat_field_corrected_rgb, correction_factor):
        intensity_corrected_rgb = flat_field_corrected_rgb / correction_factor
    else:
        raise ValueError("Not able to perform intensity correction")

    return intensity_corrected_rgb
