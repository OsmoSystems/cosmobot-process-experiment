import os
from typing import Tuple, Dict, Union

import pandas as pd
import numpy as np

from osmo_camera import raw, tiff
from osmo_camera.file_structure import create_output_directory
from osmo_camera.select_ROI import get_ROIs_for_image
from osmo_camera.stats.main import roi_statistic_calculators
from osmo_camera.correction.main import correct_image


def get_ROI_statistics(ROI):
    channel_stats = {
        stat_name: stat_calculator(ROI)
        for stat_name, stat_calculator in roi_statistic_calculators.items()
    }

    flattened_channel_stats = {
        f"{color}_{stat}": channel_stats[stat][color_index]
        for stat in channel_stats
        for color_index, color in enumerate(
            "rgb"
        )  # TODO: is it a safe assumption that everything's in rgb??
    }

    return flattened_channel_stats


def save_ROI_crops(ROI_crops_dir, raw_image_path, rgb_ROIs_by_name):
    """ Save ROI crops from the given image as .PNGs in the specified directory

    Args:
        ROI_crops_dir: The directory where ROI crops should be saved
        raw_image_path: The full path of the image to save ROI crops from
        rgb_ROIs_by_name: A map of {ROI_name: rgb_ROI}, where rgb_ROI is an `RGB Image`

    Returns:
        None
    """
    # Construct ROI crop file name from root filename plus ROI name, plus .png extension
    image_filename_root, _ = os.path.splitext(os.path.basename(raw_image_path))
    for ROI_name, rgb_ROI in rgb_ROIs_by_name.items():
        ROI_crop_filename = f"ROI {ROI_name} - {image_filename_root}.tiff"
        ROI_crop_path = os.path.join(ROI_crops_dir, ROI_crop_filename)
        tiff.save.as_tiff(rgb_ROI, ROI_crop_path)


def process_ROIs(rgb_image, raw_image_path, ROI_definitions, ROI_crops_dir=None):
    """ Process all the ROIs in a single JPEG+RAW image into summary statistics

    For each ROI:
        1. Crop
        2. Calculate summary stats
        3. Optionally, if `ROI_crops_dir` is provided, save ROI crop as a .PNG in that directory

    Args:
        rgb_image: 3d numpy array representing an image.
        raw_image_path: The full file path of a JPEG+RAW image.
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
        ROI_crops_dir: Optional. If provided, ROI crops will be saved to this directory as .PNGs

    Returns:
        An array of summary statistics dictionaries - one for each ROI
    """
    ROIs = get_ROIs_for_image(rgb_image, ROI_definitions)

    if ROI_crops_dir is not None:
        save_ROI_crops(ROI_crops_dir, raw_image_path, ROIs)

    exif_tags = raw.metadata.get_exif_tags(raw_image_path)

    return pd.DataFrame(
        [
            pd.Series(
                {
                    "timestamp": exif_tags.capture_datetime,
                    "iso": exif_tags.iso,
                    "exposure_seconds": exif_tags.exposure_time,
                    "image": os.path.basename(raw_image_path),
                    "ROI": ROI_name,
                    "ROI definition": ROI_definitions[ROI_name],
                    **get_ROI_statistics(ROI),
                }
            )
            for ROI_name, ROI in ROIs.items()
        ]
    )


def process_image(
    original_rgb_image: np.ndarray,
    original_image_filepath: str,
    ROI_definitions: Dict[str, Tuple],
    raw_images_dir: str,
    flat_field_filepath_or_none: Union[str, None],
    save_ROIs: bool = False,
    save_dark_frame_corrected_image: bool = False,
    save_flat_field_corrected_image: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """ Process an image by applying corrections and analyzing ROIs

    Args:
        original_rgb_image: RGB image to process
        original_image_filepath: path to where the image is stored on disk
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
            Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI:
            (start_col, start_row, cols, rows)
        raw_images_dir: The directory where the original raw images live
        flat_field_filepath_or_none: The image to use for flat field correction, or None to skip flat field correction
        save_ROIs: Optional. If True, ROIs will be saved as .TIFFs in a new subdirectory of raw_images_dir
        save_dark_frame_corrected_image: whether to save the dark frame corrected image to disk
        save_flat_field_corrected_image: whether to save the flat field corrected image to disk

    Returns:
        2-tuple of: roi_statistics, image_diagnostics
            roi_statistics is a pd.DataFrame, each row of which contains summary statistics for a single ROI in a
            single image.
            image_diagnostics is a pandas Series of diagnostics for an entire image; the name of this series is the
            image filename.
    """

    # If ROI crops should be saved, create a directory for them
    ROI_crops_dir = None
    if save_ROIs:
        ROI_crops_dir = create_output_directory(raw_images_dir, "ROI crops")
        print("ROI crops saved in:", ROI_crops_dir)

    corrected_rgb_image, image_diagnostics = correct_image(
        original_rgb_image,
        original_image_filepath=original_image_filepath,
        flat_field_filepath_or_none=flat_field_filepath_or_none,
        save_dark_frame_corrected_image=save_dark_frame_corrected_image,
        save_flat_field_corrected_image=save_flat_field_corrected_image,
    )

    roi_statistics = process_ROIs(
        corrected_rgb_image, original_image_filepath, ROI_definitions, ROI_crops_dir
    )

    # Reorder columns to put the most commonly used ones up front
    initial_column_order = ["ROI", "image", "exposure_seconds", "iso"]
    reordered_columns = initial_column_order + [
        column for column in roi_statistics if column not in initial_column_order
    ]
    roi_statistics_ordered = roi_statistics[reordered_columns]

    return roi_statistics_ordered, image_diagnostics
