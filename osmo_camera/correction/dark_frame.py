def _can_correct_with_dark_frame(input_rgb, dark_frame_rgb):
    return input_rgb.shape() == dark_frame_rgb.shape()

def dark_frame_correction(input_rgb, dark_frame_rgb):
    if _can_correct_with_dark_frame(input_rgb, dark_frame_rgb):
        dark_frame_corrected_rgb = input_rgb - dark_frame_rgb
    else:
        raise ValueError("Not able to perform dark frame correction")

    return dark_frame_corrected_rgb
