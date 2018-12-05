from datetime import datetime

from osmo_camera.s3 import sync_from_s3
from osmo_camera.process_images import process_images
from osmo_camera.select_ROI import prompt_for_ROI_selection, draw_ROIs_on_image
from osmo_camera.summary_images import generate_summary_images
from osmo_camera.file_structure import iso_datetime_for_filename, get_files_with_extension
from osmo_camera import raw, jupyter


def _get_first_image(rgb_images_by_filepath):
    first_filepath = sorted(rgb_images_by_filepath.keys())[0]  # Assumes images are prefixed with iso-ish datetimes
    return rgb_images_by_filepath[first_filepath]


def _save_summary_statistics_csv(experiment_dir, image_summary_data):
    csv_name = f'{experiment_dir} - summary statistics (generated {iso_datetime_for_filename(datetime.now())}).csv'
    image_summary_data.to_csv(csv_name, index=False)
    print(f'Summary statistics saved as: {csv_name}\n')

    return csv_name


def _get_rgb_images_by_filepath(raw_images_directory):
    raw_image_paths = get_files_with_extension(raw_images_directory, '.jpeg')
    return {
        raw_image_path: raw.open.as_rgb(raw_image_path)
        for raw_image_path in raw_image_paths
    }


def process_experiment(
    experiment_dir,
    local_sync_directory_path,
    ROI_definitions=[],
    sync_downsample_ratio=1,
    sync_start_time=None,
    sync_end_time=None,
    save_summary_images=False,
    save_ROIs=False
):
    ''' Process all images from an experiment:
        1. Sync raw images from s3
        2. Open JPEG+RAW files as RGB images
        3. Select ROIs (if not provided)
        4. (Optional) Save summary images
        5. Process images into summary statistics...

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_directory_path: The path to the local directory where images will be synced and processed
        ROI_definitions: Optional. Pre-selected ROI_definitions: a map of {ROI_name: ROI_definition}
            Where ROI_definition is a 4-tuple in the format provided by cv2.selectROI:
                (start_col, start_row, cols, rows)
            If not provided, we'll present you with a GUI to select ROI definitions.
        sync_downsample_ratio: Optional. Ratio to downsample images by when syncing:
            If downsample_ratio = 1, keep all images (default)
            If downsample_ratio = 2, keep half of the images for each variant
            If downsample_ratio = 3, keep one in three images
        sync_start_time: Optional. If provided, no images before this datetime will by synced
        sync_end_time: Optional. If provided, no images after this datetime will by synced
        save_summary_images: Optional. If True, ROIs will be saved as .PNGs in a new subdirectory of
            local_sync_directory_path
        save_ROIs: Optional. If True, ROIs will be saved as .PNGs in a new subdirectory of local_sync_directory_path

    Returns:
        image_summary_data: A pandas DataFrame of summary statistics
        ROI_definitions: The ROI definitions used in the processing

        Saves the image_summary_data as a .csv in the directory where this function was called.
    '''
    print(f'1. Sync images from s3 to local directory within {local_sync_directory_path}...')
    raw_images_dir = sync_from_s3(
        experiment_dir,
        local_sync_directory_path=local_sync_directory_path,
        downsample_ratio=sync_downsample_ratio,
        start_time=sync_start_time,
        end_time=sync_end_time,
    )

    print('2. Open all JPEG+RAW images as RGB images...')
    rgb_images_by_filepath = _get_rgb_images_by_filepath(raw_images_dir)

    # Display the first image for reference
    first_rgb_image = _get_first_image(rgb_images_by_filepath)
    jupyter.show_image(first_rgb_image, title='Reference image', figsize=[7, 7])

    print('3. Prompt for ROI selections (if not provided)...')
    if not ROI_definitions:
        ROI_definitions = prompt_for_ROI_selection(first_rgb_image)
        print('ROI definitions:', ROI_definitions)

    jupyter.show_image(
        draw_ROIs_on_image(first_rgb_image, ROI_definitions),
        title='Reference image with labelled ROIs',
        figsize=[7, 7]
    )

    saving_or_not = 'Save' if save_summary_images else 'Don\'t save'
    print(f'4. {saving_or_not} summary images...')
    if save_summary_images:
        generate_summary_images(rgb_images_by_filepath, ROI_definitions, raw_images_dir)

    print('5. Process images into summary statistics...')
    image_summary_data = process_images(rgb_images_by_filepath, ROI_definitions, raw_images_dir, save_ROIs)
    _save_summary_statistics_csv(experiment_dir, image_summary_data)

    return image_summary_data, ROI_definitions
