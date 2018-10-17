import math
import os

from osmo_camera import rgb, dng
from osmo_camera.file_structure import get_files_with_extension, create_output_directory
from osmo_camera.select_ROI import draw_ROIs_on_image


def generate_summary_images(raw_images_dir, ROI_definitions):
    ''' Pick some representative images and draw ROIs on them for reference

    Args:
        raw_images_dir: The directory of images to process
        ROI_definitions: Definitions of Regions of Interest (ROIs) to summarize. A map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)

    Returns:
        The name of the directory where the summary images are saved
    '''
    dng_image_paths = get_files_with_extension(raw_images_dir, '.dng')

    # Create a new directory where these images will be saved
    summary_images_dir = create_output_directory(raw_images_dir, 'summary images')

    # Pick a representative sample of images
    sample_image_paths = [
        dng_image_paths[0],  # First
        dng_image_paths[math.floor(len(dng_image_paths) / 2)],  # Middle
        dng_image_paths[-1],  # Last
    ]

    # Draw ROIs on them and save
    for image_path in sample_image_paths:
        rgb_image = dng.open.as_rgb(image_path)
        rgb_image_with_ROIs = draw_ROIs_on_image(rgb_image, ROI_definitions)

        # Save in new directory, with same name but as a .png
        filename_root, extension = os.path.splitext(os.path.basename(image_path))
        summary_image_path = os.path.join(summary_images_dir, f'{filename_root}.png')

        rgb.save.as_file(rgb_image_with_ROIs, summary_image_path)

        print(f'Summary images saved in: {summary_images_dir}\n')
        return summary_images_dir
