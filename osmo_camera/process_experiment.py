import os

from osmo_camera.s3_sync import sync_images_from_s3
from osmo_camera.process_images import process_images
from osmo_camera.select_ROI import prompt_for_ROI_selection
from osmo_camera.raw.convert import convert_all_raw_to_dng
from osmo_camera import dng, jupyter


# TODO: optional flag for whether to output cropped images
# TODO: make a dropdown for `experiment_dir`?
def process_experiment(experiment_dir, local_sync_dir, raspi_raw_location, ROI_definitions=[]):
    ''' Process all images from an experiment:
        1. Sync images from s3
        2. Select ROIs
        3. Process all ROIs on all images

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_dir: TBD
        raspi_raw_location: TBD
        ROI_definitions: pre-selected ROI definition(s), optional

    Returns:
        image_summary_data: A pandas DataFrame of summary statistics
        ROI_definitions: The ROI definitions used in the processing
    '''
    # 1. Sync images from s3 to local tmp folder
    raw_images_dir = sync_images_from_s3(experiment_dir, local_sync_dir)

    # 2. Convert all images from raw to dng
    convert_all_raw_to_dng(raw_images_dir, raspi_raw_location)

    dng_image_paths = [
        os.path.join(raw_images_dir, filename)
        for filename in os.listdir(raw_images_dir)
        if filename.endswith('.dng')  # TODO: don't save .dngs alongside jpegs?
    ]

    first_dng_image_path = dng_image_paths[0]

    first_rgb_image = dng.open.as_rgb(first_dng_image_path)

    print('First image:')
    jupyter.show_image(first_rgb_image)

    # 3. Prompt for ROI selection (if not provided)
    if not ROI_definitions:
        ROIs = prompt_for_ROI_selection(first_rgb_image)

        # TODO: prompt to name ROIs instead of numbering them
        ROI_definitions = {
            f'ROI_{index}': ROI
            for index, ROI in enumerate(ROIs)
        }

    # 4. Process images into a single DataFrame of summary statistics - one row for each ROI in each image
    image_summary_data = process_images(raw_images_dir, raspi_raw_location, ROI_definitions)

    # Output:
    #   summary image(s) # TODO: implement
    #   ROI_definition(s)
    #   csv of data (make it extensible - think about path to adding new columns to this) # TODO: save to csv
    #   optional: html file with cropped images embedded & labeled # TODO: implement
    return image_summary_data, ROI_definitions
