from subprocess import call


# TODO: tmp folder name should be deterministic so it is idempotent?
def sync_images_from_s3(experiment_name, local_sync_dir):
    ''' Syncs raw images from s3 to a local tmp directory (can optionally be provided)

    Args:
        experiment_name: The name of the experiment directory in s3
        local_sync_dir: The full path of the directory to sync locally

    Returns:
        Full path of the tmp directory for this experiment
    '''

    # if not local_dir:
    #     local_dir = '/Users/jaime/osmo/cosmobot-data-sets'  # TODO: use tmp directory instead of hardcoding mine

    sync_folder_location = f'{local_sync_dir}/{experiment_name}'  # TODO: use os path

    # TODO: use boto?
    command = f'aws s3 sync s3://camera-sensor-experiments/{experiment_name} {sync_folder_location}'
    call([command], shell=True)

    return sync_folder_location
