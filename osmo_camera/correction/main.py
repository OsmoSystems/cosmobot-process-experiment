from typing import Union, Tuple

import numpy as np
import pandas as pd

from osmo_camera.correction import dark_frame, flat_field
from osmo_camera import tiff, file_structure

DARK_FRAME_PREFIX = 'dark_frame_'
FLAT_FIELD_PREFIX = 'flat_field_'


def save_rgb_image_with_suffix(
    rgb_image,
    image_filepath,
    filepath_suffix
):
    tiff.save.as_tiff(
        rgb_image,
        file_structure.replace_extension(
            file_structure.append_suffix_to_filepath_before_extension(image_filepath, filepath_suffix),
            '.tiff'
        )
    )


def correct_image(
    original_rgb_image: np.ndarray,
    original_image_filepath: str,
    flat_field_filepath_or_none: Union[str, None],
    save_dark_frame_corrected_image: bool,
    save_flat_field_corrected_image: bool,
) -> Tuple[np.ndarray, pd.Series]:
    ''' Correct an RGB image from an experiment:
        1. Apply dark frame correction
        2. Apply flat field correction
    Also perform diagnostics on each step, raising warnings if key parameters are out of the expected range

    Args:
        original_rgb_image: RGB image to be processed
        original_image_filepath: filesystem path to the raw image
        flat_field_filepath_or_none: flat field image to use, or None if flat fielding is to be skipped
        save_dark_frame_corrected_image: whether to save the dark frame corrected image to disk
        save_flat_field_corrected_image: whether to save the flat field corrected image to disk
    Returns:
        2-tuple of:
            corrected rgb image
            A pandas Series of diagnostic information, with its name from original_image_filepath
    '''
    dark_frame_corrected_rgb_image = dark_frame.get_metadata_and_apply_dark_frame_correction(
        original_rgb_image,
        original_image_filepath,
    )
    dark_frame_diagnostics = dark_frame.get_dark_frame_diagnostics(
        original_rgb_image,
        dark_frame_corrected_rgb_image,
        original_image_filepath,
    )

    if save_dark_frame_corrected_image:
        save_rgb_image_with_suffix(dark_frame_corrected_rgb_image, original_image_filepath, '_dark_adj')

    flat_field_corrected_rgb_image = flat_field.load_flat_field_and_apply_correction(
        dark_frame_corrected_rgb_image,
        flat_field_filepath_or_none
    )
    flat_field_diagnostics = flat_field.get_flat_field_diagnostics(
        dark_frame_corrected_rgb_image,
        flat_field_corrected_rgb_image,
        original_image_filepath,
    )

    if save_flat_field_corrected_image:
        save_rgb_image_with_suffix(flat_field_corrected_rgb_image, original_image_filepath, '_dark_flat_adj')

    all_diagnostics = pd.concat(
        [
            dark_frame_diagnostics.add_prefix(DARK_FRAME_PREFIX),
            flat_field_diagnostics.add_prefix(FLAT_FIELD_PREFIX),
        ]
    )
    all_diagnostics.name = original_image_filepath
    return flat_field_corrected_rgb_image, all_diagnostics
