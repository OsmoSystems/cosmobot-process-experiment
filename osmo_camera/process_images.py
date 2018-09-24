from itertools import chain
import os

import pandas as pd

from osmo_camera import dng, raw, rgb


def _process_roi(roi_crop):
    channel_stats = {
        'mean': rgb.stats.get_channel_means(roi_crop),
        'median': rgb.stats.get_channel_medians(roi_crop),
        'min': rgb.stats.get_channel_minimums(roi_crop),
        'max': rgb.stats.get_channel_maximums(roi_crop),
        'stdev': rgb.stats.get_channel_stdevs(roi_crop),
    }

    flattened_channel_stats = {
        f'{color}_{stat}': channel_stats[stat][color_index]
        for stat in channel_stats
        for color_index, color in enumerate('rgb')  # TODO: is it a safe assumption that everything's in rgb??
    }

    return flattened_channel_stats


def process_image(raw_image_path, raspiraw_location, ROIs):
    ''' Process all the ROIs in a single image into summary statistics

    1. Convert JPEG+RAW -> .DNG
    2. For each ROI:
        a. Crop
        b. Calculate summary stats

    Args:
        raw_image_path: The full file path of the JPEG+RAW image from a RaspberryPi camera.
        ROIs: Regions of Interest (ROIs) to crop and summarize

    Returns:
        An array of summary statistics dictionaries - one for each ROI
    '''

    dng_image_path = raw.convert.to_dng(raspiraw_location, raw_image_path=raw_image_path)
    rgb_image = dng.open.as_rgb(dng_image_path)

    ROI_crops = {
        roi_name: rgb.image_basics.crop_image(rgb_image, roi_definition)
        for roi_name, roi_definition in ROIs.items()
    }

    return [
        {
            'timestamp': dng.metadata.capture_date(raw_image_path),
            'image': os.path.basename(raw_image_path),
            'ROI': roi_name,
            **_process_roi(roi_crop)
        }
        for roi_name, roi_crop in ROI_crops.items()
    ]


def process_images(raw_images_dir, raspiraw_location, ROIs):
    ''' Process all images in a given directory

    Args:
        raw_images_dir: The directory of images to process
        ROIs: Regions of Interest (ROIs) to crop and summarize

    Returns:
        An pandas DataFrame in which each row contains summary statistics for a single ROI in a single image
    '''

    raw_image_paths = [
        os.path.join(raw_images_dir, filename)
        for filename in os.listdir(raw_images_dir)
        if filename.endswith('.jpeg')
    ]

    # Generate "summary" images: a few representative full images with outlines of ROIs selected
    # Just generates and save them in the current folder
    # generate_summary_images()

    summary_statistics = pd.DataFrame(
        # Flatten all ROI summaries for all images into a single 1D list
        list(chain.from_iterable([
            process_image(raw_image_path, raspiraw_location, ROIs)
            for raw_image_path in raw_image_paths
        ]))
    ).sort_values('timestamp').reset_index(drop=True)

    return summary_statistics
