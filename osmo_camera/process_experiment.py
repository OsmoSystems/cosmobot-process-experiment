from osmo_camera.s3 import sync_from_s3
from osmo_camera.process_images import process_images
from osmo_camera.select_ROI import prompt_for_ROI_selection
from osmo_camera.summary_images import generate_summary_images
from osmo_camera.directories import get_files_with_extension
from osmo_camera import raw, dng, jupyter


def _open_first_image(raw_images_dir):
    # Assumes images have already been converted to .DNGs
    dng_image_paths = get_files_with_extension(raw_images_dir, '.dng')

    first_dng_image_path = dng_image_paths[0]
    first_rgb_image = dng.open.as_rgb(first_dng_image_path)

    return first_rgb_image


def process_experiment(
    experiment_dir,
    raspiraw_location,
    ROI_definitions=[],
    local_sync_dir=None,
    save_summary_images=False,
    save_ROIs=False
):
    ''' Process all images from an experiment:
        1. Sync raw images from s3
        2. Convert raw images to .DNG
        3. Select ROIs (if not provided)
        4. Process all ROIs on all images

    Args:
        experiment_dir: The name of the experiment directory in s3
        raspiraw_location: The name of the local directory where raspiraw is installed
        local_sync_dir: Optional. The name of the local directory where images will be synced and processed
        ROI_definitions: Optional. Pre-selected ROI_definitions: a map of {ROI_name: ROI_definition}
        Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
        save_summary_images: Optional. If True, ROIs will be saved as .PNGs in a new subdirectory of local_sync_dir
        save_ROIs: Optional. If True, ROIs will be saved as .PNGs in a new subdirectory of local_sync_dir

    Returns:
        image_summary_data: A pandas DataFrame of summary statistics
        ROI_definitions: The ROI definitions used in the processing

        Saves the image_summary_data as a .csv in the directory where this function was called.
    '''
    print('1. Sync images from s3 to local tmp directory...')
    raw_images_dir = sync_from_s3(experiment_dir, local_sync_dir)

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

    saving_or_not = 'Save' if save_summary_images else 'Don\'t save'
    print(f'4. {saving_or_not} summary images...')
    if save_summary_images:
        summary_images_dir = generate_summary_images(raw_images_dir, ROI_definitions)
        print(f'Summary images saved in: {summary_images_dir}\n')

    print('5. Process images into summary statistics...')
    image_summary_data = process_images(raw_images_dir, ROI_definitions, save_ROIs)

    csv_name = f'{experiment_dir} - summary statistics.csv'
    image_summary_data.to_csv(csv_name, index=False)
    print(f'Summary statistics saved as: {csv_name}\n')

    return image_summary_data, ROI_definitions
