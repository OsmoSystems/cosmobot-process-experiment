from osmo_camera import rgb

filepath_suffix = "_dark_flat_intensity_adj"


def _apply_intensity_correction(input_rgb, ROI_definition_for_intensity_correction):
    # TODO (https://app.asana.com/0/862697519982053/933995774091561): implement
    return input_rgb


def apply_intensity_correction_to_rgb_images(
    flat_field_corrected_rgb_by_filepath,
    ROI_definition_for_intensity_correction,
    save_corrected_images=False
):
    intensity_corrected_rgb_by_filepath = {
        image_path: _apply_intensity_correction(
            flat_field_corrected_rgb,
            ROI_definition_for_intensity_correction
        )
        for image_path, flat_field_corrected_rgb in flat_field_corrected_rgb_by_filepath.items()
    }

    # TODO: save with each correction or batch save after all corrections are applied?
    if save_corrected_images:
        rgb.save.save_rgb_images_by_filepath_with_suffix(intensity_corrected_rgb_by_filepath, filepath_suffix)

    return intensity_corrected_rgb_by_filepath
