def _apply_intensity_correction(input_rgb, ROI_definition_for_intensity_correction):
    # TODO (https://app.asana.com/0/862697519982053/933995774091561): implement
    return input_rgb


def apply_intensity_correction_to_rgb_images(
    flat_field_corrected_rgb_by_filepath,
    ROI_definition_for_intensity_correction
):
    intensity_corrected_rgb_by_filepath = {
        image_path: _apply_intensity_correction(
            flat_field_corrected_rgb,
            ROI_definition_for_intensity_correction
        )
        for image_path, flat_field_corrected_rgb in flat_field_corrected_rgb_by_filepath.items()
    }

    return intensity_corrected_rgb_by_filepath
