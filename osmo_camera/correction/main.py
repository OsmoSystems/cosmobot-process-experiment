from osmo_camera.correction import dark_frame, flat_field, intensity
from osmo_camera import rgb, file_structure


def save_rgb_images_by_filepath_with_suffix(
    rgb_images_by_filepath,
    filepath_suffix,
    save_all_images
):
    images_to_save = rgb_images_by_filepath if save_all_images else first_middle_last_rgb_images(rgb_images_by_filepath)

    for image_path, image_rgb in images_to_save.items():
        if not file_structure.file_exists_and_size_is_not_zero(image_path):
            rgb.save.as_file(
                image_rgb,
                file_structure.append_suffix_to_filepath_before_extension(image_path, filepath_suffix)
            )


def first_middle_last_rgb_images(rgb_images_by_filepath):
    sorted_rgb_image_keys = sorted(rgb_images_by_filepath.keys())  # Assumes images are prefixed with iso-ish datetimes
    keys_to_use = [
        sorted_rgb_image_keys[0],
        sorted_rgb_image_keys[int(len(sorted_rgb_image_keys) / 2)],
        sorted_rgb_image_keys[-1]
    ]

    return {
        image_path: rgb_images_by_filepath[image_path]
        for image_path in keys_to_use
    }


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

    print('Saving dark frame corrected images')
    save_rgb_images_by_filepath_with_suffix(
        dark_frame_corrected_rgb_by_filepath,
        "_dark_adj",
        save_dark_frame_corrected_images
    )

    print('2. Apply flat field correction, but not really')
    flat_field_corrected_rgb_by_filepath = flat_field.apply_flat_field_correction_to_rgb_images(
        dark_frame_corrected_rgb_by_filepath
    )

    print('Saving flat field corrected images')
    save_rgb_images_by_filepath_with_suffix(
        flat_field_corrected_rgb_by_filepath,
        "_dark_flat_adj",
        save_flat_field_corrected_images
    )

    print('3. Apply intensity correction, but not really')
    intensity_corrected_rgb_by_filepath = intensity.apply_intensity_correction_to_rgb_images(
        flat_field_corrected_rgb_by_filepath,
        ROI_definition_for_intensity_correction
    )

    # TODO: save with each correction or batch save after all corrections are applied?
    print('Saving intensity corrected images')
    save_rgb_images_by_filepath_with_suffix(
        intensity_corrected_rgb_by_filepath,
        "_dark_flat_intensity_adj",
        save_intensity_corrected_images
    )

    return intensity_corrected_rgb_by_filepath