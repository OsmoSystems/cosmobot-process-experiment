from typing import List, Dict

import math
import os

import imageio
import numpy as np
from PIL import Image

from osmo_camera import tiff, raw
from osmo_camera.file_structure import create_output_directory, get_files_with_extension
from osmo_camera.select_ROI import draw_ROIs_on_image


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
        rgb_image_with_ROIs = draw_ROIs_on_image(rgb_image, ROI_definitions)

        # Save in new directory, with same name but as a .png.
        filename_root, extension = os.path.splitext(os.path.basename(image_path))
        summary_image_path = os.path.join(summary_images_dir, f'{filename_root}.tiff')

        tiff.save.as_tiff(rgb_image_with_ROIs, summary_image_path)

        print(f'Summary images saved in: {summary_images_dir}\n')
    return summary_images_dir


def _read_all_filenames(experiment_directories, local_sync_directory_path):
    all_filenames = []
    for experiment_directory in experiment_directories:
        filenames = get_files_with_extension(os.path.join(local_sync_directory_path, experiment_directory), '.jpeg')
        all_filenames.append(filenames)

    return [filename for sublist in all_filenames for filename in sublist]


def generate_summary_gif(
    experiment_directories,
    local_sync_directory_path,
    ROI_definitions,
    name='summary',
    image_resize_factor=5,
):
    output_filename = f'{name}.gif'
    image_dimensions = (3280, 2464)

    all_filenames_flattened = _read_all_filenames(experiment_directories, local_sync_directory_path)

    images = []
    for filename in all_filenames_flattened:
        rgb_image = raw.open.as_rgb(filename)
        annotated_image = draw_ROIs_on_image(rgb_image, ROI_definitions)
        PIL_image = Image.fromarray((annotated_image * 255).astype('uint8'))
        resized_PIL_image = PIL_image.resize((
            round(image_dimensions[0]/image_resize_factor),
            round(image_dimensions[1]/image_resize_factor)
        ))
        resized_numpy_image = np.array(resized_PIL_image)
        images.append(resized_numpy_image)
    imageio.mimsave(output_filename, images)
    return output_filename


def generate_summary_video(experiment_directories, local_sync_directory_path, ROI_definitions, name='summary'):
    output_filename = f'{name}.mp4'
    writer = imageio.get_writer(output_filename, fps=1)

    all_filenames_flattened = _read_all_filenames(experiment_directories, local_sync_directory_path)

    for filename in all_filenames_flattened:
        rgb_image = raw.open.as_rgb(filename)
        annotated_image = draw_ROIs_on_image(rgb_image, ROI_definitions)
        rgb_image = (annotated_image * 255).astype('uint8')
        writer.append_data(rgb_image)

    writer.close()
    return output_filename
