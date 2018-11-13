import platform
import os
import tempfile
from subprocess import check_call
import re
import boto


def sync_from_s3(experiment_directory_name, local_sync_dir=None):
    ''' Syncs raw images from s3 to a local tmp directory (can optionally be provided)

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_dir: The full path of the directory to sync locally

    Returns:
        Full path of the tmp directory for this experiment
    '''
    if not local_sync_dir:
        # On MacOS (Darwin), tempfile.gettempdir() returns a weird auto-generated directory
        # e.g. '/var/folders/nj/269977hs0_96bttwj2gs_jhhp48z54/T'
        # https://stackoverflow.com/questions/847850/cross-platform-way-of-getting-temp-directory-in-python
        local_sync_dir = '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()

    sync_directory_location = os.path.join(local_sync_dir, experiment_directory_name)

    # Would be better to use boto, but neither boto nor boto3 support sync
    # https://github.com/boto/boto3/issues/358
    command = f'aws s3 sync s3://camera-sensor-experiments/{experiment_directory_name} {sync_directory_location}'
    check_call([command], shell=True)

    return sync_directory_location


def _filter_and_sort_experiment_list(experiment_names, regex):
    # Filter with a regex and sort the list of directories to sort most recent first
    filtered_list = [experiment_name for experiment_name in experiment_names if re.search(regex, experiment_name)]
    return sorted(filtered_list, reverse=True)


def _order_experiment_list_by_isodate_formats(experiment_names):
    experiment_names_with_hyphens_in_isodate = _filter_and_sort_experiment_list(
        experiment_names,
        r'^\d{4}-\d\d-\d\d.'
    )
    experiment_names_without_hyphens_in_isodate = _filter_and_sort_experiment_list(experiment_names, r'^\d{8}.')
    return experiment_names_with_hyphens_in_isodate + experiment_names_without_hyphens_in_isodate


def list_experiments():
    ''' Lists all experiment directories in the "camera-sensor-experiments" bucket

        Returns: a list of experiment names that is filtered and ordered (by isodate formats YYYY-MM-DD & YYYYMMDD)
        The list will be a concatenated set of lists, with the items starting with a list of YYYY-MM-DD formated names
        that are ordered by descending date followed by the same ordering but with a list of YYYYMMDD formatted names.
        Any files or folders that do not match these two formats will be discarded.
    '''
    try:
        # TODO (SOFT-538): Stop checking in access key!
        s3 = boto.connect_s3('AKIAIFJ2IMOKIWPKGZRA', 'vqTb5DpoSouOtgmTJo+Zm8+mtW9KeddRxbFeliny')
    except boto.exception.NoAuthHandlerFound:
        print('You must have aws credentials already saved, e.g. via `aws configure`. \n')
        return []

    bucket = s3.get_bucket('camera-sensor-experiments')
    experiment_directories = bucket.list('', '/')

    experiment_names = [directory.name.strip('/') for directory in experiment_directories]
    return _order_experiment_list_by_isodate_formats(experiment_names)
