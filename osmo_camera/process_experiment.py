from osmo_camera.s3_sync import sync_images_from_s3
from osmo_camera.process_images import process_images
from osmo_camera.select_ROI import prompt_for_ROI_selection


# TODO: optional flag for whether to output cropped images
# TODO: make a dropdown for `experiment_dir`?
def process_experiment(experiment_dir, ROIs=[]):
    ''' Process all images from an experiment:
        1. Sync images from s3
        2. Select ROIs
        3. Process all ROIs on all images

    Args:
        experiment_dir: The name of the experiment directory in s3
        ROIs: pre-selected ROI(s), optional

    Returns:
        A pandas DataFrame of summary statistics
    '''
    # 1. Sync images from s3 to local tmp folder
    raw_images_dir = sync_images_from_s3(experiment_dir)  # TODO: implement

    # 2. Prompt for ROI selection (if not provided)
    if not ROIs:
        image = None  # TODO: pick an image to use for ROI selection. Convert to .DNG first?
        ROIs = prompt_for_ROI_selection(image)  # TODO: implement

    # 3. Process images into a single DataFrame of summary statistics - one row for each ROI in each image
    image_summary_data = process_images(raw_images_dir, ROIs)

    # Output:
    #   summary image(s) # TODO: implement
    #   ROI(s)  # TODO: return
    #   csv of data (make it extensible - think about path to adding new columns to this) # TODO: save to csv
    #   optional: html file with cropped images embedded & labeled # TODO: implement
    return image_summary_data
