import warnings

import pandas as pd

from osmo_camera.correction import dark_frame, diagnostics, flat_field
from osmo_camera import tiff, file_structure


DARK_FRAME_PREFIX = 'dark_frame_'
FLAT_FIELD_PREFIX = 'flat_field_'


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
    flat_field_filepath,
    save_dark_frame_corrected_images,
    save_flat_field_corrected_images,
):
    ''' Correct all images from an experiment:
        1. Apply dark frame correction
        2. Apply flat field correction

    Args:
        original_rgb_by_filepath: A Series of RGB images indexed by raw file path
    Returns:
        2-tuple of:
            A Series of corrected rgb images indexed by raw file path
            A DataFrame of diagnostic information indexed by raw file path
    '''
    dark_frame_corrected_rgb_by_filepath = dark_frame.apply_dark_frame_correction_to_rgb_images(
        original_rgb_by_filepath
    )
    dark_frame_diagnostics = diagnostics.run_diagnostics(
        original_rgb_by_filepath,
        dark_frame_corrected_rgb_by_filepath,
        dark_frame.get_dark_frame_diagnostics
    )

    if save_dark_frame_corrected_images:
        save_rgb_images_by_filepath_with_suffix(dark_frame_corrected_rgb_by_filepath, "_dark_adj")

    if flat_field_filepath is None:
        warnings.warn('No `flat_field_filepath` provided. Flat field correction *not* applied')
        flat_field_corrected_rgb_by_filepath = dark_frame_corrected_rgb_by_filepath
        flat_field_diagnostics = pd.DataFrame()
    else:
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
        save_rgb_images_by_filepath_with_suffix(flat_field_corrected_rgb_by_filepath, "_dark_flat_adj")

    all_diagnostics = pd.concat(
        [
            dark_frame_diagnostics.add_prefix(DARK_FRAME_PREFIX),
            flat_field_diagnostics.add_prefix(FLAT_FIELD_PREFIX),
        ],
        axis='columns'
    )
    return flat_field_corrected_rgb_by_filepath, all_diagnostics
