from osmo_camera.correction import dark_frame, flat_field, intensity
from osmo_camera import rgb, file_structure


def save_rgb_images_by_filepath_with_suffix(
    rgb_images_by_filepath,
    filepath_suffix
):
    for image_path, image_rgb in rgb_images_by_filepath.items():
        rgb.save.as_uint16_tiff(
            image_rgb,
            file_structure.replace_extension(
                file_structure.append_suffix_to_filepath_before_extension(image_path, filepath_suffix),
                '.tiff'
            )
        )


def correct_images(
    original_rgb_by_filepath,
    ROI_definition_for_intensity_correction,
    save_dark_frame_corrected_images,
    save_flat_field_corrected_images,
    save_intensity_corrected_images
):
    ''' Correct all images from an experiment:
        1. Apply dark frame correction
        2. Apply flat field correction
        3. Apply intensity correction

    Args:
        original_rgb_by_filepath: A map of {image_path: rgb_image}
        ROI_definition_for_intensity_correction: ROI to average and use to correct intensity on `ROI_definitions`
     Returns:
        A dictionary of intensity corrected rgb images that is keyed by raw file path
    '''

    print('--------Correcting Images--------')
    print('1. Apply dark frame correction')
    dark_frame_corrected_rgb_by_filepath = dark_frame.apply_dark_frame_correction_to_rgb_images(
        original_rgb_by_filepath
    )

    if save_dark_frame_corrected_images:
        print('Saving dark frame corrected images')
        save_rgb_images_by_filepath_with_suffix(dark_frame_corrected_rgb_by_filepath, "_dark_adj")

    print('2. Apply flat field correction, but not really')
    flat_field_corrected_rgb_by_filepath = flat_field.apply_flat_field_correction_to_rgb_images(
        dark_frame_corrected_rgb_by_filepath
    )

    if save_flat_field_corrected_images:
        print('Saving flat field corrected images')
        save_rgb_images_by_filepath_with_suffix(flat_field_corrected_rgb_by_filepath, "_dark_flat_adj")

    print('3. Apply intensity correction, but not really')
    intensity_corrected_rgb_by_filepath = intensity.apply_intensity_correction_to_rgb_images(
        flat_field_corrected_rgb_by_filepath,
        ROI_definition_for_intensity_correction
    )

    if save_intensity_corrected_images:
        print('Saving intensity corrected images')
        save_rgb_images_by_filepath_with_suffix(intensity_corrected_rgb_by_filepath, "_dark_flat_intensity_adj")

    return intensity_corrected_rgb_by_filepath
