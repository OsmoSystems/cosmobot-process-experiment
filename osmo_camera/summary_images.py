import math
import os

from osmo_camera import rgb
from osmo_camera.file_structure import create_output_directory
from osmo_camera.select_ROI import draw_ROIs_on_image


def generate_summary_images(rgb_images_by_filepath, ROI_definitions, raw_images_dir):
    ''' Pick some representative images and draw ROIs on them for reference

    Args:
        rgb_images_by_filepath: A map of {image_path: rgb_image}
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
        raw_images_dir: The directory of images to process

    Returns:
        The name of the directory where the summary images are saved
    '''
    summary_images_dir = create_output_directory(raw_images_dir, 'summary images')

    # Pick a representative sample of images (assumes images are prefixed with iso-ish datetimes)
    rgb_image_paths = sorted(rgb_images_by_filepath.keys())
    sample_image_paths = [
        rgb_image_paths[0],  # First
        rgb_image_paths[math.floor(len(rgb_image_paths) / 2)],  # Middle
        rgb_image_paths[-1],  # Last
    ]

    # Draw ROIs on them and save
    for image_path in sample_image_paths:
        rgb_image = rgb_images_by_filepath[image_path]
        rgb_image_with_ROIs = draw_ROIs_on_image(rgb_image, ROI_definitions)

        # Save in new directory, with same name but as a .png.
        filename_root, extension = os.path.splitext(os.path.basename(image_path))
        summary_image_path = os.path.join(summary_images_dir, f'{filename_root}.tiff')

        rgb.save.as_uint32_tiff(rgb_image_with_ROIs, summary_image_path)

        print(f'Summary images saved in: {summary_images_dir}\n')
        return summary_images_dir
