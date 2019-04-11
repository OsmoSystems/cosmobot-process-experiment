from datetime import datetime
import os

import pandas as pd
import tqdm

from osmo_camera.s3 import sync_from_s3
from osmo_camera.process_images import process_images
from osmo_camera.select_ROI import prompt_for_ROI_selection, draw_ROIs_on_image
from osmo_camera.summary_images import generate_summary_images
from osmo_camera.file_structure import iso_datetime_for_filename, get_files_with_extension
from osmo_camera import raw, jupyter


def _open_first_image(raw_image_paths):
    first_filepath = sorted(raw_image_paths)[0]  # Assumes images are prefixed with iso-ish datetimes
    return raw.open.as_rgb(first_filepath)


def _save_summary_statistics_csv(experiment_dir, image_summary_data):
    csv_name = f'{experiment_dir} - summary statistics (generated {iso_datetime_for_filename(datetime.now())}).csv'
    image_summary_data.to_csv(csv_name, index=False)
    print(f'Summary statistics saved as: {csv_name}\n')

    return csv_name


def get_raw_image_paths_for_experiment(local_sync_directory_path, experiment_directory):
    ''' Opens all JPEG+RAW images in the specified experiment directory and returns as a map of
        {image_filepath: `RGB Image`}.

        A convenience function intended to be used by technicians inside a jupyter notebook, which will
        already have `local_sync_directory` and `experiment_directory` as variables.

    Args:
        local_sync_directory_path: The path to the local directory where images will be synced and processed
        experiment_directory: The name of the experiment directory (the folder inside the local_sync_directory that you
        want to open images from)

    Return:
        A pandas Series of {image_filepath: `RGB Image`}
    '''
    raw_images_directory = os.path.join(local_sync_directory_path, experiment_directory)
    raw_image_paths = get_files_with_extension(raw_images_directory, '.jpeg')
    return pd.Series(raw_image_paths)


def open_and_process_images(
        experiment_dir,
        raw_images_dir,
        raw_image_paths,
        ROI_definitions,
        flat_field_filepath=None,
        save_summary_images=False,
        save_ROIs=False,
        save_dark_frame_corrected_images=False,
        save_flat_field_corrected_images=False,
):
    rgb_images_by_filepath = pd.Series({
        raw_image_path: raw.open.as_rgb(raw_image_path)
        for raw_image_path in raw_image_paths
    })
    if save_summary_images:
        generate_summary_images(rgb_images_by_filepath, ROI_definitions, raw_images_dir)

    roi_summary_data, image_diagnostics = process_images(
        rgb_images_by_filepath,
        ROI_definitions,
        raw_images_dir,
        flat_field_filepath,
        save_ROIs=save_ROIs,
        save_dark_frame_corrected_images=save_dark_frame_corrected_images,
        save_flat_field_corrected_images=save_flat_field_corrected_images,
    )

    return roi_summary_data, image_diagnostics


def process_experiment(
    experiment_dir,
    local_sync_directory_path,
    ROI_definitions=[],
    flat_field_filepath=None,
    sync_downsample_ratio=1,
    sync_start_time=None,
    sync_end_time=None,
    save_summary_images=False,
    save_ROIs=False,
    save_dark_frame_corrected_images=False,
    save_flat_field_corrected_images=False,
):
    ''' Process all images from an experiment:
        1. Sync raw images from s3
        2. Open JPEG+RAW files as RGB images
        3. Select ROIs (if not provided)
        4. (Optional) Save summary images
        5. Process images into summary statistics...

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_directory_path: The path to the local directory where images will be synced and processed
        flat_field_filepath: The path of the image to use for flat field correction. Must be a .npy file.
        ROI_definitions: Optional. Pre-selected ROI_definitions: a map of {ROI_name: ROI_definition}
            Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI:
                (start_col, start_row, cols, rows)
            If not provided, we'll present you with a GUI to select ROI definitions.
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
        Raises warnings if any of the image diagnostics are outside of normal ranges.
    '''
    print(f'1. Sync images from s3 to local directory within {local_sync_directory_path}...')
    raw_images_dir = sync_from_s3(
        experiment_dir,
        local_sync_directory_path=local_sync_directory_path,
        downsample_ratio=sync_downsample_ratio,
        start_time=sync_start_time,
        end_time=sync_end_time,
    )

    raw_image_paths = get_raw_image_paths_for_experiment(local_sync_directory_path, experiment_dir)

    # Display the first image for reference
    first_rgb_image = _open_first_image(raw_image_paths)

    print('2. Prompt for ROI selections (if not provided)...')
    if not ROI_definitions:
        ROI_definitions = prompt_for_ROI_selection(first_rgb_image)
        print('ROI definitions:', ROI_definitions)

    jupyter.show_image(
        draw_ROIs_on_image(first_rgb_image, ROI_definitions),
        title='Reference image with labelled ROIs',
        figsize=[7, 7]
    )

    saving_or_not = 'save' if save_summary_images else 'don\'t save'

    print(f'3. Process images into summary statistics and {saving_or_not} summary images...')

    roi_summary_data_and_image_diagnostics_dfs_for_files = [
        # Returns roi_summary_data df, image_diagnostics df -> resulting list will be a list of 2-tuples
        open_and_process_images(
            experiment_dir=experiment_dir,
            raw_images_dir=raw_images_dir,
            raw_image_paths=[raw_image_path],  # Hack: Process in "batches" of 1 image to avoid big refactor.
            ROI_definitions=ROI_definitions,
            flat_field_filepath=flat_field_filepath,
            save_summary_images=save_summary_images,
            save_ROIs=save_ROIs,
            save_dark_frame_corrected_images=save_dark_frame_corrected_images,
            save_flat_field_corrected_images=save_flat_field_corrected_images,
        )
        # tqdm_notebook is the tqdm progress bar version for use in jupyter notebooks
        for raw_image_path in tqdm.tqdm_notebook(raw_image_paths)
    ]

    roi_summary_data_for_files, image_diagnostics_for_files = zip(*roi_summary_data_and_image_diagnostics_dfs_for_files)

    roi_summary_data_for_all_files = pd.concat(roi_summary_data_for_files).reset_index(drop=True)
    image_diagnostics_for_all_files = pd.concat(image_diagnostics_for_files).reset_index(drop=True)

    _save_summary_statistics_csv(experiment_dir, roi_summary_data_for_all_files)

    return roi_summary_data_for_all_files, image_diagnostics_for_all_files, ROI_definitions
