from itertools import chain
from functools import partial
import os
from typing import Callable, Dict

import numpy as np
import pandas as pd

from osmo_camera import dng, rgb
from osmo_camera.correction import dark_frame, flat_field, intensity
from osmo_camera.file_structure import create_output_directory, get_files_with_extension
from osmo_camera.select_ROI import get_ROIs_for_image


# Running numpy calculations against this axis aggregates over the image for each channel, as color channels are axis=2
IMAGE_AXES = (0, 1)


def _coefficient_of_variation(image):
    return np.std(image, axis=IMAGE_AXES) / np.mean(image, axis=IMAGE_AXES)


# Type annotation clears things up for Mypy
ROI_STATISTIC_CALCULATORS: Dict[str, Callable] = {
    'mean': partial(np.mean, axis=IMAGE_AXES),
    'median': partial(np.median, axis=IMAGE_AXES),
    'min': partial(np.amin, axis=IMAGE_AXES),
    'max': partial(np.amax, axis=IMAGE_AXES),
    'stdev': partial(np.std, axis=IMAGE_AXES),
    'cv': _coefficient_of_variation,
    **{
        f'percentile_{percentile}': partial(np.percentile, q=percentile, axis=IMAGE_AXES)
        for percentile in [99, 95, 90, 75, 50, 25]
    }
}


def _process_ROI(ROI):
    channel_stats = {
        stat_name: stat_calculator(ROI)
        for stat_name, stat_calculator in ROI_STATISTIC_CALCULATORS.items()
    }

    flattened_channel_stats = {
        f'{color}_{stat}': channel_stats[stat][color_index]
        for stat in channel_stats
        for color_index, color in enumerate('rgb')  # TODO: is it a safe assumption that everything's in rgb??
    }

    return flattened_channel_stats


def save_ROI_crops(ROI_crops_dir, dng_image_path, rgb_ROIs_by_name):
    ''' Save ROI crops from the given image as .PNGs in the specified directory

    Args:
        ROI_crops_dir: The directory where ROI crops should be saved
        dng_image_path: The full path of the image to save ROI crops from
        rgb_ROIs_by_name: A map of {ROI_name: rgb_ROI}, where rgb_ROI is an `RGB Image`

    Returns:
        None
    '''
    # Construct ROI crop file name from root filename plus ROI name, plus .png extension
    image_filename_root, _ = os.path.splitext(os.path.basename(dng_image_path))
    for ROI_name, rgb_ROI in rgb_ROIs_by_name.items():
        ROI_crop_filename = f'ROI {ROI_name} - {image_filename_root}.png'
        ROI_crop_path = os.path.join(ROI_crops_dir, ROI_crop_filename)
        rgb.save.as_file(rgb_ROI, ROI_crop_path)


def ROI_analysis(rgb_image, dng_image_path, ROI_definitions, ROI_crops_dir=None):
    ''' Process all the ROIs in a single .DNG image into summary statistics

    For each ROI:
        1. Crop
        2. Calculate summary stats
        3. Optionally, if `ROI_crops_dir` is provided, save ROI crop as a .PNG in that directory

    Args:
        rgb_image: 3d numpy array representing an image.
        dng_image_path: The full file path of a DNG image.
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
        ROI_crops_dir: Optional. If provided, ROI crops will be saved to this directory as .PNGs

    Returns:
        An array of summary statistics dictionaries - one for each ROI
    '''
    ROIs = get_ROIs_for_image(rgb_image, ROI_definitions)

    if ROI_crops_dir is not None:
        save_ROI_crops(ROI_crops_dir, dng_image_path, ROIs)

    exif_tags = dng.metadata.get_exif_tags(dng_image_path)

    return [
        {
            'timestamp': exif_tags.capture_datetime,
            'iso': exif_tags.iso,
            'exposure_seconds': exif_tags.exposure_time,
            'image': os.path.basename(dng_image_path),
            'ROI': ROI_name,
            'ROI definition': ROI_definitions[ROI_name],
            **_process_ROI(ROI)
        }
        for ROI_name, ROI in ROIs.items()
    ]


def correct_images(
    dng_image_paths,
    ROI_definitions=[],
    ROI_for_intensity_correction=(0,0,0,0),
):
    ''' Process all images from an experiment:
        1. Apply dark frame correction
        2. Apply flat field correction
        3. Apply intensity correction

    Args:
        dng_image_paths: list of file paths for dngs to be processed into a "blue" value
        ROI_definitions: list of ROIs that are averaged for
        ROI_for_intensity_correction: region to average and use to correct intensity on `ROI_definitions`
     Returns:
        A dictionary of intensity corrected rgb images that is keyed by dng file path
    '''

    # Image corrections and blueness calculation
    dark_frame_corrected_rgb_by_filepath = dict()
    flat_field_corrected_rgb_by_filepath = dict()
    intensity_corrected_rgb_by_filepath = dict()

    print('1. Apply dark frame correction')
    # open all images and perform dark frame correction
    for image_path in dng_image_paths:
        image_rgb = dng.open.as_rgb(image_path)
        dark_frame_rgb = np.zeros(image_rgb.shape)
        dark_frame_corrected_rgb_by_filepath[image_path] = dark_frame.apply_dark_frame_correction(
            image_rgb,
            dark_frame_rgb
        )

    print('2. Apply flat field correction')
    for image_path in dark_frame_corrected_rgb_by_filepath:
        dark_frame_corrected_rgb = np.array(dark_frame_corrected_rgb_by_filepath[image_path])
        dark_frame_rgb = np.zeros(dark_frame_corrected_rgb.shape)
        flat_field_rgb = np.zeros(dark_frame_corrected_rgb.shape)

        flat_field_corrected_rgb_by_filepath[image_path] = flat_field.apply_flat_field_correction(
            dark_frame_corrected_rgb,
            dark_frame_rgb,
            flat_field_rgb
        )

    print('3. Apply intensity correction')
    for image_path in flat_field_corrected_rgb_by_filepath:
        intensity_correction_roi_spatial_average = rgb.average.spatial_average_of_roi(
            flat_field_corrected_rgb_by_filepath[image_path],
            ROI_for_intensity_correction
        )

        intensity_corrected_rgb_by_filepath[image_path] = intensity.apply_intensity_correction(
            flat_field_corrected_rgb_by_filepath[image_path],
            intensity_correction_roi_spatial_average
        )

    return intensity_corrected_rgb_by_filepath


def process_images(dng_images_dir, ROI_definitions, save_ROIs=False):
    ''' Process all images in a given directory

    Args:
        dng_images_dir: The directory of images to process. Assumes images have already been converted to .DNGs
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
        save_ROIs: Optional. If True, ROIs will be saved as .PNGs in a new subdirectory of dng_images_dir

    Returns:
        An pandas DataFrame in which each row contains summary statistics for a single ROI in a single image
    '''
    dng_image_paths = get_files_with_extension(dng_images_dir, '.dng')

    # If ROI crops should be saved, create a directory for them
    ROI_crops_dir = None
    if save_ROIs:
        ROI_crops_dir = create_output_directory(dng_images_dir, 'ROI crops')
        print('ROI crops saved in:', ROI_crops_dir)

    dummy_intensity_correction_ROI = (0,0,0,0)

    intensity_corrected_rgb_images = correct_images(
        dng_image_paths,
        ROI_definitions,
        ROI_for_intensity_correction=dummy_intensity_correction_ROI
    )

    processed_images = [
        ROI_analysis(rgb_image, dng_image_path, ROI_definitions, ROI_crops_dir)
        for dng_image_path, rgb_image in intensity_corrected_rgb_images.items()
    ]

    summary_statistics = pd.DataFrame(
        # Flatten all ROI summaries for all images into a single 1D list
        list(chain.from_iterable(processed_images))
    ).sort_values('timestamp').reset_index(drop=True)

    return summary_statistics
