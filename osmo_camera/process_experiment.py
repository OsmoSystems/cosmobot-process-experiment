import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import os
from typing import List, Tuple, Dict, Union

import pandas as pd
from tqdm.auto import tqdm

from osmo_camera.s3 import sync_from_s3
from osmo_camera.process_image import process_image
from osmo_camera.summary_images import generate_summary_images
from osmo_camera.file_structure import (
    iso_datetime_for_filename,
    get_files_with_extension,
)
from osmo_camera import raw, rgb, jupyter


def _open_first_image(raw_image_paths):
    # Assumes images are prefixed with iso-ish datetimes
    first_filepath = sorted(raw_image_paths)[0]
    return raw.open.as_rgb(first_filepath)


def save_summary_statistics_csv(
    experiment_name, roi_summary_data, save_directory_path: str = ""
):
    """ Saves summary statistics as a csv file in the current directory and returns the output filename.

    Args:
        experiment_name: The experiment name to use in the output filename.
        roi_summary_data: The image ROI summary data DataFrames to be exported to csv.
            This data is returned by process_experiment.
        save_directory_path: A full or relative path to the directory where the csv should be saved.
            Defaults to current working directory.

    Return:
        A string with the filename of the saved csv file.
    """
    # Create directories on the path if they don't already exist
    Path(save_directory_path).mkdir(parents=True, exist_ok=True)

    csv_filename = f"{experiment_name} - summary statistics (generated {iso_datetime_for_filename(datetime.now())}).csv"
    csv_filepath = Path(save_directory_path) / csv_filename
    roi_summary_data.to_csv(csv_filepath, index=False)
    print(f"Summary statistics saved to: {csv_filepath}\n")

    return csv_filepath


def get_raw_image_paths_for_experiment(local_sync_directory_path, experiment_directory):
    """ Opens all JPEG+RAW images in the specified experiment directory and returns as a map of
        {image_filepath: `RGB Image`}.

        A convenience function intended to be used by technicians inside a jupyter notebook, which will
        already have `local_sync_directory` and `experiment_directory` as variables.

    Args:
        local_sync_directory_path: The path to the local directory where images will be synced and processed
        experiment_directory: The name of the experiment directory (the folder inside the local_sync_directory that you
        want to open images from)

    Return:
        A pandas Series of {image_filepath: `RGB Image`}
    """
    raw_images_directory = os.path.join(local_sync_directory_path, experiment_directory)
    raw_image_paths = get_files_with_extension(raw_images_directory, ".jpeg")
    return pd.Series(raw_image_paths)


def _stack_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """ stack pandas DataFrames logically into a bigger DataFrame,
    resets the index of the resulting DataFrame to avoid duplicates in the index
    """
    return pd.concat(dataframes).reset_index(drop=True)


def _stack_serieses(serieses: List[pd.Series]) -> pd.DataFrame:
    """ stack pandas Series logically into a DataFrame
    Args:
        serieses: iterable of Pandas series

    Returns:
        pandas DataFrame with a row per series. If each Series has a Name, that will be its index label
    """
    return pd.concat(serieses, axis="columns").T


def _process_images(
    raw_image_paths: pd.Series,
    raw_images_dir: str,
    ROI_definitions: Dict[str, Tuple],
    flat_field_filepath_or_none: Union[str, None],
    save_ROIs: bool,
    save_dark_frame_corrected_images: bool,
    save_flat_field_corrected_images: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Process a full set of images, with parallelization if multiple CPU threads are available on this machine
    """

    def _process_image_local(raw_image_path):
        """ Version of process_image with all of the local configuration variables packed in.
        Also encapsulates the opening of the image.
        """
        return process_image(
            original_rgb_image=raw.open.as_rgb(raw_image_path),
            original_image_filepath=raw_image_path,
            raw_images_dir=raw_images_dir,
            ROI_definitions=ROI_definitions,
            flat_field_filepath_or_none=flat_field_filepath_or_none,
            save_ROIs=save_ROIs,
            save_dark_frame_corrected_image=save_dark_frame_corrected_images,
            save_flat_field_corrected_image=save_flat_field_corrected_images,
        )

    with ThreadPoolExecutor() as executor:
        # We want identical warnings to be shown only for the first image they occur on (the default),
        # but we also want subsequent calls to process_experiment to start with a fresh warning store
        # so that warnings don't stop showing after the first run.
        # catch_warnings gives us this fresh warning store.
        with warnings.catch_warnings():
            # process_image returns roi_summary_data df, image_diagnostics df -> this will be a list of 2-tuples
            roi_summary_data_and_image_diagnostics_dfs_for_files = list(
                tqdm(
                    executor.map(_process_image_local, raw_image_paths),
                    total=len(raw_image_paths),
                )
            )
    roi_summary_data_for_files, image_diagnostics_for_files = zip(
        *roi_summary_data_and_image_diagnostics_dfs_for_files
    )

    roi_summary_data_for_all_files = _stack_dataframes(roi_summary_data_for_files)
    image_diagnostics_for_all_files = _stack_serieses(image_diagnostics_for_files)

    return roi_summary_data_for_all_files, image_diagnostics_for_all_files


def process_experiment(
    experiment_dir,
    local_sync_directory_path,
    ROI_definitions=None,
    flat_field_filepath=None,
    sync_downsample_ratio=1,
    sync_start_time=None,
    sync_end_time=None,
    save_summary_images=False,
    save_ROIs=False,
    save_dark_frame_corrected_images=False,
    save_flat_field_corrected_images=False,
):
    """ Process all images from an experiment:
        1. Sync raw images from s3
        2. Prompt for ROI selections (using first image) if ROI_definitions not provided
        3. Process images into summary statistics and save summary images

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_directory_path: The path to the local directory where images will be synced and processed
        flat_field_filepath: The path of the image to use for flat field correction. Must be a .npy file.
        ROI_definitions: Pre-selected ROI_definitions: a map of {ROI_name: ROI_definition}
            Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI:
                (start_col, start_row, cols, rows)
        sync_downsample_ratio: Optional. Ratio to downsample images by when syncing:
            If downsample_ratio = 1, keep all images (default)
            If downsample_ratio = 2, keep half of the images for each variant
            If downsample_ratio = 3, keep one in three images
        sync_start_time: Optional. If provided, no images before this datetime will by synced
        sync_end_time: Optional. If provided, no images after this datetime will by synced
        save_summary_images: Optional. If True, ROIs will be saved as .PNGs in a new subdirectory of
            local_sync_directory_path
        save_ROIs: Optional. If True, ROIs will be saved as .TIFFs in a new subdirectory of local_sync_directory_path
        save_dark_frame_corrected_images: Optional. If True, dark-frame-corrected images will be saved as .TIFFs with a
            `_dark_adj` suffix
        save_flat_field_corrected_images: Optional. If True, flat-field-corrected images will be saved as .TIFFs with a
            `_dark_flat_adj` suffix

    Returns:
        roi_summary_data: pandas DataFrame of summary statistics of ROIs
        image_diagnostics: pandas DataFrame of diagnostic information on images through the correction process
            Documentation of individual diagnostics and warnings is in README.md in the project root.
        ROI_definitions: The ROI definitions used in the processing

    Side effects:
        Saves the roi_summary_data as a .csv in the directory where this function was called.
        Raises warnings if any of the image diagnostics are outside of normal ranges. If multiple images have matching
        diagnostic warnings, only one copy of a particular warning will be shown.
    """
    if not ROI_definitions:
        raise ValueError(
            "No ROI definitions. Cannot process images.\n"
            "If you'd just like to download images, use osmo_camera.s3.sync_from_s3().\n"
            "For ROI selection, see osmo_camera.select_ROI.ROISelectionInterface."
        )

    print(
        f"1. Sync images from s3 to local directory within {local_sync_directory_path}..."
    )
    raw_images_dir = sync_from_s3(
        experiment_dir,
        local_sync_directory_path=local_sync_directory_path,
        downsample_ratio=sync_downsample_ratio,
        start_time=sync_start_time,
        end_time=sync_end_time,
    )

    raw_image_paths = get_raw_image_paths_for_experiment(
        local_sync_directory_path, experiment_dir
    )

    # Display the first image for reference
    first_rgb_image = _open_first_image(raw_image_paths)
    jupyter.show_image(
        rgb.annotate.draw_ROIs_on_image(first_rgb_image, ROI_definitions),
        title="Reference image with labelled ROIs",
        figsize=[7, 7],
    )

    saving_or_not = "save" if save_summary_images else "don't save"
    print(
        f"2. Process images into summary statistics and {saving_or_not} summary images..."
    )
    if save_summary_images:
        generate_summary_images(raw_image_paths, ROI_definitions, raw_images_dir)

    roi_summary_data_for_all_files, image_diagnostics_for_all_files = _process_images(
        raw_image_paths,
        raw_images_dir=raw_images_dir,
        ROI_definitions=ROI_definitions,
        flat_field_filepath_or_none=flat_field_filepath,
        save_ROIs=save_ROIs,
        save_dark_frame_corrected_images=save_dark_frame_corrected_images,
        save_flat_field_corrected_images=save_flat_field_corrected_images,
    )

    return (
        roi_summary_data_for_all_files,
        image_diagnostics_for_all_files,
        ROI_definitions,
    )
