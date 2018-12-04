from osmo_camera import rgb

def apply_intensity_correction(input_rgb, ROI_definition_for_intensity_correction):
    intensity_correction_roi_spatial_average = rgb.average.spatial_average_of_roi(  # pylint: disable=W0612
        input_rgb,
        ROI_definition_for_intensity_correction
    )

    return input_rgb
