import pandas as pd

from osmo_camera.correction import dark_frame, diagnostics, flat_field
from osmo_camera import tiff, file_structure


DARK_FRAME_PREFIX = 'dark_frame_'
FLAT_FIELD_PREFIX = 'flat_field_'
INTENSITY_ADJ_PREFIX = 'intensity_adj_'


def save_rgb_images_by_filepath_with_suffix(
    rgb_images_by_filepath,
    filepath_suffix
):
    for image_path, image_rgb in rgb_images_by_filepath.items():
        tiff.save.as_tiff(
            image_rgb,
            file_structure.replace_extension(
                file_structure.append_suffix_to_filepath_before_extension(image_path, filepath_suffix),
                '.tiff'
            )
        )


def correct_images(
    original_rgb_by_filepath,
    ROI_definition_for_intensity_correction,
    flat_field_filepath,
    save_dark_frame_corrected_images,
    save_flat_field_corrected_images,
    save_intensity_corrected_images
):
    ''' Correct all images from an experiment:
        1. Apply dark frame correction
        2. Apply flat field correction
        3. Apply intensity correction

    Args:
        original_rgb_by_filepath: A Series of RGB images indexed by raw file path
        ROI_definition_for_intensity_correction: ROI to average and use to correct intensity on `ROI_definitions`
    Returns:
        2-tuple of:
            A Series of corrected rgb images indexed by raw file path
            A DataFrame of diagnostic information indexed by raw file path
    '''

    print('--------Correcting Images--------')
    print('1. Apply dark frame correction')
    dark_frame_corrected_rgb_by_filepath = dark_frame.apply_dark_frame_correction_to_rgb_images(
        original_rgb_by_filepath
    )
    dark_frame_diagnostics = diagnostics.run_diagnostics(
        original_rgb_by_filepath,
        dark_frame_corrected_rgb_by_filepath,
        dark_frame.get_dark_frame_diagnostics
    )

    if save_dark_frame_corrected_images:
        print('Saving dark frame corrected images')
        save_rgb_images_by_filepath_with_suffix(dark_frame_corrected_rgb_by_filepath, "_dark_adj")

    print('2. Apply flat field correction')
    flat_field_corrected_rgb_by_filepath = flat_field.apply_flat_field_correction_to_rgb_images(
        dark_frame_corrected_rgb_by_filepath,
        flat_field_filepath
    )
    flat_field_diagnostics = diagnostics.run_diagnostics(
        dark_frame_corrected_rgb_by_filepath,
        flat_field_corrected_rgb_by_filepath,
        flat_field.get_flat_field_diagnostics
    )

    if save_flat_field_corrected_images:
        print('Saving flat field corrected images')
        save_rgb_images_by_filepath_with_suffix(flat_field_corrected_rgb_by_filepath, "_dark_flat_adj")

    print('3. Apply intensity correction, but not really')
    # For now, bypass all iteration to save time and memory
    intensity_corrected_rgb_by_filepath = flat_field_corrected_rgb_by_filepath
    intensity_correction_diagnostics = pd.DataFrame({})
    # TODO (https://app.asana.com/0/862697519982053/933995774091561): actually implement
    # intensity_corrected_rgb_by_filepath = intensity.apply_intensity_correction_to_rgb_images(
    #     flat_field_corrected_rgb_by_filepath,
    #     ROI_definition_for_intensity_correction
    # )
    # intensity_correction_diagnostics = diagnostics.run_diagnostics(
    #     flat_field_corrected_rgb_by_filepath,
    #     intensity_corrected_rgb_by_filepath,
    #     intensity.get_intensity_correction_diagnostics
    # )

    if save_intensity_corrected_images:
        print('Saving intensity corrected images')
        save_rgb_images_by_filepath_with_suffix(intensity_corrected_rgb_by_filepath, "_dark_flat_intensity_adj")

    all_diagnostics = pd.concat(
        [
            dark_frame_diagnostics.add_prefix(DARK_FRAME_PREFIX),
            flat_field_diagnostics.add_prefix(FLAT_FIELD_PREFIX),
            intensity_correction_diagnostics.add_prefix(INTENSITY_ADJ_PREFIX),
        ],
        axis='columns'
    )
    return intensity_corrected_rgb_by_filepath, all_diagnostics
