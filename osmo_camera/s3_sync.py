# TODO: implement actual sync, instead of relying on it already being done
def sync_images_from_s3(experiment_dir):
    ''' Syncs raw images from s3 to a local tmp directory

    Args:
        experiment_dir: The name of the experiment directory in s3
    Returns:
        Name of the tmp directory
    '''

    # For now, bypass this step by doing it manually
    raw_image_tmp_folder = '/Users/jaime/osmo/cosmobot-data-set-subset'

    return raw_image_tmp_folder
