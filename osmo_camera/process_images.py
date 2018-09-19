from itertools import chain
import os

import pandas as pd

from osmo_camera import raw_dng, image_stats, image_basics


def _convert_to_dng(raw_image_filename):
    # TODO: actually do the conversion instead of assuming converted file already exists
    # TODO: correctly manage camera v1 vs v2?
    filename_prefix, file_extension = os.path.splitext(raw_image_filename)
    return f'{filename_prefix}.dng'


def _process_roi(roi_crop):
    channel_stats = {
        'mean': image_stats.get_channel_means(roi_crop),
        'median': image_stats.get_channel_medians(roi_crop),
        'min': image_stats.get_channel_minimums(roi_crop),
        'max': image_stats.get_channel_maximums(roi_crop),
        'stdev': image_stats.get_channel_stdevs(roi_crop),
    }

    flattened_channel_stats = {
        f'{color}_{stat}': channel_stats[stat][color_index]
        for stat in channel_stats
        for color_index, color in enumerate('bgr')
    }

    return flattened_channel_stats


def process_image(raw_image_filename, ROIs):
    # For each image:
        # RAW -> .DNG (does this depend on camera v1 vs v2?)
        # For each ROI:
            # Crop
            # Calculate summary stats

    dng_image_filename = _convert_to_dng(raw_image_filename)
    dng_image = raw_dng.open_image(dng_image_filename)

    ROI_crops = {
        roi_name: image_basics.crop_image(dng_image, roi_definition)
        for roi_name, roi_definition in ROIs.items()
    }

    return [
        {
            'timestamp': raw_dng.get_create_date(raw_image_filename),
            'image': os.path.basename(raw_image_filename),
            'ROI': roi_name,
            **_process_roi(roi_crop)
        }
        for roi_name, roi_crop in ROI_crops.items()
    ]


def process_images(raw_images_dir, ROIs):
    image_filenames = [
        os.path.join(raw_images_dir, file)
        for file in os.listdir(raw_images_dir)
        if file.endswith('.jpeg')  # TODO: don't save .dngs alongside jpegs?
    ]

    # Generate "summary" images: a few representative full images with outlines of ROIs selected
    # Just generates and save them in the current folder
    # generate_summary_images() # TODO: implement

    # For each image:
    #   Convert RAW -> .DNG
    #   For each ROI:
    #       Crop image to just that ROI
    #       Calculate summary stats
    image_summary_data = pd.DataFrame(
        list(chain.from_iterable([
            process_image(image_filename, ROIs)
            for image_filename in image_filenames
        ]))
    ).sort_values('timestamp').reset_index(drop=True)

    return image_summary_data


def generate_raw_to_dng_conversion_command(source_files_folder, raspi_raw_location):
    raspi_raw_command = os.path.join(raspi_raw_location, 'raspiraw/raspi_dng_ov')

    for file in os.listdir(source_files_folder):
        input_file = os.path.join(source_files_folder, file)

        file_basename, dot, extension = file.partition('.')
        if extension != 'jpeg':
            # This is not one of our files
            continue
        output_filename = f'{file_basename}.dng'
        output_file = os.path.join(source_files_folder, 'dng', output_filename)

        print(raspi_raw_command, f'"{input_file}"', f'"{output_file}"')


if __name__ == '__main__':
    raspi_raw_location = '/Users/jaime/osmo'
    source_files_folder = '/Users/jaime/osmo/cosmobot-data-set-subset'
    generate_raw_to_dng_conversion_command(source_files_folder, raspi_raw_location)
