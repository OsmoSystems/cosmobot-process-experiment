from osmo_camera.s3_sync import sync_images_from_s3
from osmo_camera.process_images import process_images


# TODO: optional flag for whether to output cropped images
# TODO: make a dropdown for `experiment_dir`?
def process_experiment(experiment_dir, ROIs):
    ''' Process images from an experiment

    Args:
        experiment_dir: The name of the experiment directory in s3
        ROIs: pre-selected ROI(s)
    Returns:
        TBD
    '''
    # 1. Sync images from s3 to local tmp folder (output name so you can delete it?)
    # tmp folder name should be deterministic so it is idempotent
    raw_images_dir = sync_images_from_s3(experiment_dir)  # TODO: implement

    # 2. Prompt for ROI selection (if not provided)
    # Require each ROI to be labelled
    # Make it easier to label "high" and "low"?
    if not ROIs:
        # Open question: do these need to be converted to .DNG first before selecting?
        ROIs = prompt_for_ROI_selection()  # TODO: implement

    # 3. Process images
    # For each image:
    #   Convert RAW -> .DNG
    #   For each ROI:
    #       Crop image to just that ROI
    #       Calculate summary stats
    image_summary_data = process_images(raw_images_dir, ROIs)

    # Output:
    #   summary image(s) # TODO: implement
    #   ROI(s)  # TODO: return
    #   csv of data (make it extensible - think about path to adding new columns to this)
    #   optional: html file with cropped images embedded & labeled # TODO: implement
    return image_summary_data
