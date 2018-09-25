import platform
import os
import tempfile
from subprocess import call
import boto


def sync_images_from_s3(experiment_dir, local_sync_dir=None):
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

    sync_folder_location = os.path.join(local_sync_dir, experiment_dir)

    # Would be better to use boto, but neither boto nor boto3 support sync
    # https://github.com/boto/boto3/issues/358
    command = f'aws s3 sync s3://camera-sensor-experiments/{experiment_dir} {sync_folder_location}'
    call([command], shell=True)

    return sync_folder_location


def list_experiments():
    # TODO: don't check in access keys! (should deactivate this one and get a new one)
    s3 = boto.connect_s3('AKIAI4OKPYSTXJ2PZNYQ', 'Nu8t0W57FLQLIQx2WfKokz4Kcu/JDOZEaez3H3WR')
    bucket = s3.get_bucket('camera-sensor-experiments')
    experiment_folders = bucket.list('', '/')

    experiment_names = [folder.name.strip('/') for folder in experiment_folders]

    # Reverse list of folders to sort most recent first (assumes folder name starts with ISO date)
    return list(reversed(experiment_names))
