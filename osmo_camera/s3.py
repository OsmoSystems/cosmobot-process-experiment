import platform
import os
import tempfile
from subprocess import call
import boto


def sync_from_s3(s3_directory, local_sync_dir=None):
    ''' Syncs raw images from s3 to a local tmp directory (can optionally be provided)

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_dir: The full path of the directory to sync locally

    Returns:
        Full path of the tmp directory for this experiment
    '''
    if not local_sync_dir:
        # On MacOS (Darwin), tempfile.gettempdir() returns a weird auto-generated folder
        # e.g. '/var/folders/nj/269977hs0_96bttwj2gs_jhhp48z54/T'
        # https://stackoverflow.com/questions/847850/cross-platform-way-of-getting-temp-directory-in-python
        local_sync_dir = '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()

    sync_folder_location = os.path.join(local_sync_dir, s3_directory)

    # Would be better to use boto, but neither boto nor boto3 support sync
    # https://github.com/boto/boto3/issues/358
    command = f'aws s3 sync s3://camera-sensor-experiments/{s3_directory} {sync_folder_location}'
    call([command], shell=True)

    return sync_folder_location


def sync_to_s3(local_sync_dir):
    ''' Syncs raw images from a local directory to the s3://camera-sensor-experiments bucket

    Args:
        local_sync_dir: The full path of the directory to sync locally

    Returns:
       None
    '''

    # Using CLI vs boto: https://github.com/boto/boto3/issues/358
    # It looks like sync is not a supported function of the python boto library
    # Work around is to use cli sync for now (requires aws cli to be installed)
    print(f'Performing sync of experiments folder: {local_sync_dir}')

    # This argument pattern issues a uni-directional sync to S3 bucket
    # https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html
    command = f'aws s3 sync {local_sync_dir} s3://camera-sensor-experiments'
    call([command], shell=True)


def list_experiments():
    ''' Lists all experiment folders in the "camera-sensor-experiments" bucket
    '''
    try:
        # TODO (SOFT-538): Stop checking in access key!
        s3 = boto.connect_s3('AKIAIFJ2IMOKIWPKGZRA', 'vqTb5DpoSouOtgmTJo+Zm8+mtW9KeddRxbFeliny')
    except boto.exception.NoAuthHandlerFound:
        print('You must have aws credentials already saved, e.g. via `aws configure`. \n')
        return []

    bucket = s3.get_bucket('camera-sensor-experiments')
    experiment_folders = bucket.list('', '/')

    experiment_names = [folder.name.strip('/') for folder in experiment_folders]

    # Reverse list of folders to sort most recent first (assumes folder name starts with ISO date)
    return list(reversed(experiment_names))
