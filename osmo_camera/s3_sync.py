from subprocess import call


# TODO: implement actual sync, instead of relying on it already being done
# tmp folder name should be deterministic so it is idempotent?
def sync_images_from_s3(experiment_name, local_dir=None):
    ''' Syncs raw images from s3 to a local tmp directory (can optionally be provided)

    Args:
        experiment_name: The name of the experiment directory in s3
        local_dir: The full path of the directory to sync locally. Defaults to a tmp folder if not provided

    Returns:
        Full path of the tmp directory for this experiment
    '''

    # # For now, bypass this step by doing it manually
    # # raw_image_tmp_folder = '/Users/jaime/osmo/cosmobot-data-set-subset'
    # raw_image_tmp_folder = f'/Users/jaime/osmo/cosmobot-data-sets/{experiment_dir}'
    #
    # return raw_image_tmp_folder

    if not local_dir:
        local_dir = '~/osmo/cosmobot-data-sets'

    command = f'aws s3 sync s3://camera-sensor-experiments/{experiment_name} {local_dir}/{experiment_name}'
    call([command], shell=True)

#
# def sync_all_images_from_s3(output_dir=None):
#     '''Runs aws s3 sync command with output folder'''
#     # Using CLI vs boto: https://github.com/boto/boto3/issues/358
#     # It looks like sync is not a supported function of the python boto library
#     # Work around is to use cli sync for now (requires aws cli to be installed)
#     print("Performing sync of output (experiments) folder")
#
#     # This argument pattern issues a uni-directional sync to S3 bucket
#     # https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html
#     command = 'aws s3 sync {} s3://camera-sensor-experiments'.format(output_dir)
#     call([command], shell=True)
