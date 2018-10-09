from osmo_camera.s3 import sync_experiment_dir
from osmo_camera.process_images import process_images
from osmo_camera.select_ROI import prompt_for_ROI_selection
from osmo_camera.summary_images import generate_summary_images
from osmo_camera.get_files import get_files_with_extension
from osmo_camera import raw, dng, jupyter


# TODO:
# Add tests for everything
# Add docstrings to everything

# Clean-up ideas
#  - Review functions in `jupyter.py`
#  - Rename `image_basics.py`?
#  - Clean up `stats.py`

# Nice-to-have:
#  - "auto-exposure" brightening for ROI selection
#  ------
#  - Don't rely on knowing the location of raspiraw installation?
#  - Better ROI labelling?
#  - Make it easier to label "high" and "low"?


def _open_first_image(raw_images_dir):
    # Assumes images have already been converted to .DNGs
    dng_image_paths = sorted(get_files_with_extension(raw_images_dir, '.dng'))

    first_dng_image_path = dng_image_paths[0]
    first_rgb_image = dng.open.as_rgb(first_dng_image_path)

    return first_rgb_image


# TODO (SOFT-511): optionally generate and save an .html file with all of the cropped images
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
        ROI_definitions: Optional. Pre-selected ROI_definitions: a map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)

    Returns:
        image_summary_data: A pandas DataFrame of summary statistics
        ROI_definitions: The ROI definitions used in the processing

        Saves the image_summary_data as a .csv in the directory where this function was called.
    '''
    print('1. Sync images from s3 to local tmp folder...')
    raw_images_dir = sync_experiment_dir(experiment_dir, local_sync_dir)

    print('2. Convert all images from raw to dng...')
    raw.convert.to_dng(raspiraw_location, raw_images_dir=raw_images_dir)

    # Open and display the first image for reference
    first_rgb_image = _open_first_image(raw_images_dir)

    jupyter.show_image(first_rgb_image, title='Reference image', figsize=[7, 7])

    print('3. Prompt for ROI selections (if not provided)...')
    if not ROI_definitions:
        ROI_definitions = prompt_for_ROI_selection(first_rgb_image)

    jupyter.show_image(
        first_rgb_image,
        title='Reference image with labelled ROIs',
        figsize=[7, 7],
        ROI_definitions=ROI_definitions
    )

    print('4. Saving summary images...')
    summary_images_dir = generate_summary_images(raw_images_dir, ROI_definitions)
    print(f'Summary images saved in: {summary_images_dir}\n')

    print('5. Process images into summary statistics...')
    image_summary_data = process_images(raw_images_dir, raspiraw_location, ROI_definitions)

    csv_name = f'{experiment_dir} - summary statistics.csv'
    image_summary_data.to_csv(csv_name)
    print(f'Summary statistics saved as: {csv_name}\n')

    return image_summary_data, ROI_definitions
