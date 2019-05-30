from typing import List, Dict

import math
import os
import logging
from itertools import chain

import imageio
import numpy as np

from osmo_camera import tiff, raw, rgb
from osmo_camera.file_structure import create_output_directory, get_files_with_extension, datetime_from_filename


def generate_summary_images(raw_image_paths: List[str], ROI_definitions: Dict[str, tuple], raw_images_dir: str) -> str:
    ''' Pick some representative images and draw ROIs on them for reference

    Args:
        raw_image_paths: A list of paths to raw image files
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
        raw_images_dir: The directory of images to process

    Returns:
        The name of the directory where the summary images are saved
    '''
    summary_images_dir = create_output_directory(raw_images_dir, 'summary images')

    # Pick a representative sample of images (assumes images are prefixed with iso-ish datetimes)
    raw_image_paths = sorted(raw_image_paths)
    sample_image_paths = [
        raw_image_paths[0],  # First
        raw_image_paths[math.floor(len(raw_image_paths) / 2)],  # Middle
        raw_image_paths[-1],  # Last
    ]

    # Draw ROIs on them and save
    for image_path in sample_image_paths:
        rgb_image = raw.open.as_rgb(image_path)
        rgb_image_with_ROIs = rgb.annotate.draw_ROIs_on_image(rgb_image, ROI_definitions)

        # Save in new directory, with same name but as a .png.
        filename_root, extension = os.path.splitext(os.path.basename(image_path))
        summary_image_path = os.path.join(summary_images_dir, f'{filename_root}.tiff')

        tiff.save.as_tiff(rgb_image_with_ROIs, summary_image_path)

        print(f'Summary images saved in: {summary_images_dir}\n')
    return summary_images_dir


def get_experiment_image_filepaths(local_sync_directory_path, experiment_directories=None):
    ''' Get a list of all .jpeg files in a list of experiment directories.

    Args:
        local_sync_directory_path: The path to the local directory where images are synced.
        experiment_directories: Optional. A list of experiment directory names to search for images. Defaults to None.
        If experiment_directories is None, search for images in local_sync_directory_path.

    Return:
        A list of paths to all .jpeg images in the provided experiment directories.
    '''
    if experiment_directories is None:
        experiment_directories = ['']
    all_filepaths = [
        get_files_with_extension(os.path.join(local_sync_directory_path, experiment_directory), '.jpeg')
        for experiment_directory in experiment_directories
    ]
    return list(chain(*all_filepaths))


def scale_image(PIL_image, image_scale_factor):
    ''' Scale a PIL image, multiplying dimensions by a given scale factor.

    Args:
        PIL_image: A PIL image to be scaled.
        image_scale_factor: The multiplier used to scale the image width and height.
    '''
    width, height = PIL_image.size
    return PIL_image.resize((
        round(width * image_scale_factor),
        round(height * image_scale_factor)
    ))


def _annotate_image(rgb_image, ROI_definitions, show_timestamp, filepath):
    image_with_ROIs = rgb.annotate.draw_ROIs_on_image(rgb_image, ROI_definitions)

    if show_timestamp:
        timestamp = datetime_from_filename(os.path.basename(filepath))
        return rgb.annotate.draw_text_on_image(image_with_ROIs, str(timestamp))

    return image_with_ROIs


def _open_filter_annotate_and_scale_image(
    filepath,
    ROI_definitions,
    image_scale_factor,
    color_channels,
    show_timestamp
):
    rgb_image = raw.open.as_rgb(filepath)
    filtered_image = rgb.filter.select_channels(rgb_image, color_channels)
    annotated_image = _annotate_image(filtered_image, ROI_definitions, show_timestamp, filepath)

    PIL_image = rgb.convert.to_PIL(annotated_image)
    scaled_image = scale_image(PIL_image, image_scale_factor)

    return np.array(scaled_image)


def generate_summary_gif(
    filepaths,
    ROI_definitions,
    name='summary',
    image_scale_factor=0.25,
    color_channels='rgb',
    show_timestamp=True
):
    ''' Compile a list of images into a summary GIF with ROI definitions overlayed.
    Saves GIF to the current working directory.

    Args:
        filepaths: List of image file names to be compiled into the GIF.
        ROI_definitions: A map of {ROI_name: ROI_definition}
            Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI:
                (start_col, start_row, cols, rows)
        name: Optional. String name of the file to be saved. Defaults to 'summary'
        image_scale_factor: Optional. Number multiplier used to scale images to adjust file size. Defaults to 1/4.
        color_channels: Optional. Lowercase string of rgb channels to show in the output image. Defaults to 'rgb'.
        show_timestamp: Optional. Boolean indicating whether to write image timestamps in output GIF
            Defaults to True.

    Returns:
        The name of the GIF file that was saved.
    '''
    output_filename = f'{name}.gif'
    images = [
        _open_filter_annotate_and_scale_image(
            filepath,
            ROI_definitions,
            image_scale_factor,
            color_channels,
            show_timestamp
        )
        for filepath in filepaths
    ]
    imageio.mimsave(output_filename, images)
    return output_filename


def generate_summary_video(
    filepaths,
    ROI_definitions,
    name='summary',
    image_scale_factor=1,
    color_channels='rgb',
    show_timestamp=True,
    fps=1
):
    ''' Compile a list of images into a summary video with ROI definitions overlayed.
    Saves video to the current working directory.

    Args:
        filepaths: List of image file names to be compiled into the video.
        ROI_definitions: A map of {ROI_name: ROI_definition}
            Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI:
                (start_col, start_row, cols, rows)
        name: Optional. String name of the file to be saved. Defaults to 'summary'
        image_scale_factor: Optional. Number multiplier used to scale images to adjust file size. Defaults to 1.
        color_channels: Optional. Lowercase string of rgb channels to show in the output image. Defaults to 'rgb'.
        show_timestamp: Optional. Boolean indicating whether to write image timestamps in output video.
            Defaults to True.
        fps: Optional. Integer video frames-per-second. Defaults to 1.

    Returns:
        The name of the summary video file that was saved
    '''
    output_filename = f'{name}.mp4'
    writer = imageio.get_writer(output_filename, fps=fps)
    # Suppress a warning message about shoehorning image dimensions into mpeg block sizes
    logger = logging.getLogger('imageio_ffmpeg')
    logger.setLevel(logging.ERROR)

    for filepath in filepaths:
        prepared_image = _open_filter_annotate_and_scale_image(
            filepath,
            ROI_definitions,
            image_scale_factor,
            color_channels,
            show_timestamp
        )
        writer.append_data(prepared_image)

    writer.close()
    logger.setLevel(logging.WARNING)

    return output_filename
