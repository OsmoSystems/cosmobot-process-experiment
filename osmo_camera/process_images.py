import os

import pandas as pd

from osmo_camera import raw, tiff
from osmo_camera.file_structure import create_output_directory
from osmo_camera.select_ROI import get_ROIs_for_image
from osmo_camera.stats.main import roi_statistic_calculators
from osmo_camera.correction.main import correct_images


def get_ROI_statistics(ROI):
    channel_stats = {
        stat_name: stat_calculator(ROI)
        for stat_name, stat_calculator in roi_statistic_calculators.items()
    }

    flattened_channel_stats = {
        f'{color}_{stat}': channel_stats[stat][color_index]
        for stat in channel_stats
        for color_index, color in enumerate('rgb')  # TODO: is it a safe assumption that everything's in rgb??
    }

    return flattened_channel_stats


def save_ROI_crops(ROI_crops_dir, raw_image_path, rgb_ROIs_by_name):
    ''' Save ROI crops from the given image as .PNGs in the specified directory

    Args:
        ROI_crops_dir: The directory where ROI crops should be saved
        raw_image_path: The full path of the image to save ROI crops from
        rgb_ROIs_by_name: A map of {ROI_name: rgb_ROI}, where rgb_ROI is an `RGB Image`

    Returns:
        None
    '''
    # Construct ROI crop file name from root filename plus ROI name, plus .png extension
    image_filename_root, _ = os.path.splitext(os.path.basename(raw_image_path))
    for ROI_name, rgb_ROI in rgb_ROIs_by_name.items():
        ROI_crop_filename = f'ROI {ROI_name} - {image_filename_root}.tiff'
        ROI_crop_path = os.path.join(ROI_crops_dir, ROI_crop_filename)
        tiff.save.as_tiff(rgb_ROI, ROI_crop_path)


def process_ROIs(rgb_image, raw_image_path, ROI_definitions, ROI_crops_dir=None):
    ''' Process all the ROIs in a single JPEG+RAW image into summary statistics

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
    '''
    ROIs = get_ROIs_for_image(rgb_image, ROI_definitions)

    if ROI_crops_dir is not None:
        save_ROI_crops(ROI_crops_dir, raw_image_path, ROIs)

    exif_tags = raw.metadata.get_exif_tags(raw_image_path)

    return pd.DataFrame([
        pd.Series({
            'timestamp': exif_tags.capture_datetime,
            'iso': exif_tags.iso,
            'exposure_seconds': exif_tags.exposure_time,
            'image': os.path.basename(raw_image_path),
            'ROI': ROI_name,
            'ROI definition': ROI_definitions[ROI_name],
            **get_ROI_statistics(ROI)
        })
        for ROI_name, ROI in ROIs.items()
    ])


def process_images(
    original_rgb_images_by_filepath,
    ROI_definitions,
    raw_images_dir,
    flat_field_filepath,
    save_ROIs=False,
    save_dark_frame_corrected_images=False,
    save_flat_field_corrected_images=False,
):
    ''' Process all images in a given directory

    Args:
        original_rgb_images_by_filepath: A map of {image_path: rgb_image}
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
        raw_images_dir: The directory where the original raw images live
        flat_field_filepath: The path of the image to use for flat field correction. Must be a .npy file.
        save_ROIs: Optional. If True, ROIs will be saved as .TIFFs in a new subdirectory of raw_images_dir

    Returns:
        An pandas DataFrame in which each row contains summary statistics for a single ROI in a single image
    '''

    # If ROI crops should be saved, create a directory for them
    ROI_crops_dir = None
    if save_ROIs:
        ROI_crops_dir = create_output_directory(raw_images_dir, 'ROI crops')
        print('ROI crops saved in:', ROI_crops_dir)

    corrected_rgb_images, image_diagnostics = correct_images(
        original_rgb_images_by_filepath,
        flat_field_filepath,
        save_dark_frame_corrected_images=save_dark_frame_corrected_images,
        save_flat_field_corrected_images=save_flat_field_corrected_images,
    )

    processed_ROIs = [
        process_ROIs(rgb_image, raw_image_path, ROI_definitions, ROI_crops_dir)
        for raw_image_path, rgb_image in corrected_rgb_images.items()
    ]

    # One big flat DF with rows from each ROI from each image
    roi_statistics = pd.concat(
        processed_ROIs
    ).sort_values('timestamp').reset_index(drop=True)

    initial_column_order = ['ROI', 'image', 'exposure_seconds', 'iso']
    reordered_columns = initial_column_order + [
        column for column in roi_statistics if column not in initial_column_order
    ]
    roi_statistics = roi_statistics[reordered_columns]

    return roi_statistics, image_diagnostics
