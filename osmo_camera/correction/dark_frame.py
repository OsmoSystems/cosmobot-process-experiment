def dark_frame_correction(input_rgb, dark_frame_rgb):
    dark_frame_corrected_rgb = input_rgb - dark_frame_rgb
    return dark_frame_corrected_rgb
