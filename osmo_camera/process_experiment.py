import os

from osmo_camera.s3_sync import sync_images_from_s3
from osmo_camera.process_images import process_images
from osmo_camera.select_ROI import prompt_for_ROI_selection
from osmo_camera import raw, dng, jupyter


# TODO:
# Add tests for everything
# Add docstrings to everything

# Clean-up ideas
#  - Review functions in `jupyter.py`
#  - Rename `image_basics.py`
#  - Clean up `stats.py`

# Nice-to-have:
#  - Use boto instead of relying on aws cli
#  - Don't rely on knowing the location of raspiraw installation?
#  - Make a dropdown for `experiment_dir`?
#  - Better ROI labelling?
#  - Make it easier to label "high" and "low"?

# Required follow-on features:
#  - Generate summary images
#  - Optionally save cropped images
#  - Save summary statistics as a .csv


# TODO: probably move this to be more generically used to open a representative image
def open_first_image(raw_images_dir):
    # Assumes images have already been converted to .DNGs
    dng_image_paths = [
        os.path.join(raw_images_dir, filename)
        for filename in os.listdir(raw_images_dir)
        if filename.endswith('.dng')
    ]

    first_dng_image_path = dng_image_paths[0]
    first_rgb_image = dng.open.as_rgb(first_dng_image_path)

    return first_rgb_image


def process_experiment(experiment_dir, raspiraw_location, ROI_definitions=[], local_sync_dir=None):
    ''' Process all images from an experiment:
        1. Sync raw images from s3
        2. Convert raw images to .DNG
        3. Select ROIs (if not provided)
        4. Process all ROIs on all images

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_dir: The name of the local directory where images will be synced and processed
        raspiraw_location: The name of the local directory where raspiraw is installed
        ROI_definitions: pre-selected ROI definition(s), optional

    Returns:
        image_summary_data: A pandas DataFrame of summary statistics
        ROI_definitions: The ROI definitions used in the processing
    '''
    # 1. Sync images from s3 to local tmp folder
    print('1. Sync images from s3 to local tmp folder...')
    raw_images_dir = sync_images_from_s3(experiment_dir, local_sync_dir)

    # 2. Convert all images from raw to dng
    print('2. Convert all images from raw to dng...')
    raw.convert.to_dng(raspiraw_location, raw_images_dir=raw_images_dir)

    # Open and display the first image for reference
    first_rgb_image = open_first_image(raw_images_dir)
    jupyter.show_image(first_rgb_image, title='Reference image (first)', figsize=[7, 7])

    # 3. Prompt for ROI selections (if not provided)
    print('3. Prompt for ROI selections (if not provided)...')
    if not ROI_definitions:
        ROI_definitions = prompt_for_ROI_selection(first_rgb_image)

    # 4. Process images into a single DataFrame of summary statistics - one row for each ROI in each image
    print('4. Process images into summary statistics...')
    image_summary_data = process_images(raw_images_dir, raspiraw_location, ROI_definitions)

    # Output:
    #   summary image(s) # TODO: implement
    #   ROI_definition(s)
    #   csv of data (make it extensible - think about path to adding new columns to this) # TODO: save to csv
    #   optional: html file with cropped images embedded & labeled # TODO: implement
    return image_summary_data, ROI_definitions
