import datetime
import os
import re
from subprocess import check_call
from typing import Sequence, List, Optional

import boto
import numpy as np
import pandas as pd

from . import file_structure


CAMERA_SENSOR_EXPERIMENTS_BUCKET_NAME = 'camera-sensor-experiments'


def _list_camera_sensor_experiments_s3_bucket_contents(directory_name: str = '') -> List[str]:
    ''' Get a list of all of the files in a logical directory off s3, within the camera sensor experiments bucket.

    Arguments:
        directory_name: prefix within our experiments bucket on s3, inclusive of trailing slash if you'd like the list
            of files within a "directory". Default is '' to get the top-level index of the bucket.

    Returns:
        list of key names under the prefix provided.
    '''
    try:
        # TODO (SOFT-538): Stop checking in access key!
        s3 = boto.connect_s3('AKIAIFJ2IMOKIWPKGZRA', 'vqTb5DpoSouOtgmTJo+Zm8+mtW9KeddRxbFeliny')
    except boto.exception.NoAuthHandlerFound:  # type: ignore
        print('You must have aws credentials already saved, e.g. via `aws configure`. \n')
        return []

    bucket = s3.get_bucket(CAMERA_SENSOR_EXPERIMENTS_BUCKET_NAME)
    keys = bucket.list(directory_name, '/')

    return list([key.name for key in keys])


def _download_s3_files(experiment_directory: str, file_names: List[str], output_directory_path: str) -> None:
    ''' Download specific filenames from within an experiment directory on s3.
    '''

    include_args = ' '.join([
        f'--include "{file_name}"'
        for file_name in file_names
    ])

    # Would be better to use boto, but neither boto nor boto3 support sync
    # https://github.com/boto/boto3/issues/358
    command = (
        f'aws s3 sync s3://{CAMERA_SENSOR_EXPERIMENTS_BUCKET_NAME}/{experiment_directory} {output_directory_path} '
        f'--exclude "*" {include_args}'
    )
    check_call([command], shell=True)


_IMAGES_INFO_COLUMNS = [
    'Timestamp',
    'variant',
    'filename',
    'capture_group'
]


def _get_timestamp_and_variant(filename):
    timestamp = file_structure.datetime_from_filename(filename)
    rest = filename[file_structure.FILENAME_TIMESTAMP_LENGTH:]
    variant, extension = os.path.splitext(rest)
    return timestamp, variant


def _get_images_info(experiment_directory: str) -> pd.DataFrame:
    ''' Organized DataFrame of of the .jpeg file contents of a directory in our bucket on s3.

    Arguments:
        experiment_directory: directory name for an experiment within our experiment results bucket on s3
    Returns:
        a DataFrame containing one row for each .jpeg file in the experiment directory. Each row has a:
            Timestamp: a datetime taken from the filename
            variant: variant portion of the .jpeg filename
            capture_group: a capture group index; see _get_capture_groups() for what that means
            filename: full filename with extension
    '''
    s3_prefix = f'{experiment_directory}/'
    all_keys = _list_camera_sensor_experiments_s3_bucket_contents(s3_prefix)
    prefix_length = len(s3_prefix)

    jpeg_filenames = [
        key[prefix_length:]
        for key in all_keys
        if key.endswith('.jpeg')
    ]
    if not jpeg_filenames:
        # Early out to avoid zero-length errors
        return pd.DataFrame(columns=_IMAGES_INFO_COLUMNS)

    timestamps_and_variants = [_get_timestamp_and_variant(filename) for filename in jpeg_filenames]
    timestamps, variants = zip(*timestamps_and_variants)

    return pd.DataFrame({
        'Timestamp': timestamps,
        'variant': variants,
        'filename': jpeg_filenames,
        'capture_group': _get_capture_groups(variants)
    }, columns=_IMAGES_INFO_COLUMNS)


def _get_capture_groups(variants: Sequence[str]) -> pd.Series:
    ''' Provide a series which groups images into logical "capture groups"
     by time.

    This implementation deals with the case of a "small" capture group in case an experiment is terminated early,
    but it does NOT deal with missing variants in the middle of an experiment. Please don't do that.

    Args:
        variants: sequence of variants that correspond to image files. Must be in chronological order.
    Returns:
        a pd.Series of capture groups corresponding to the "loop index" that each image was taken in.
        This Series is intended to be appended onto the images_info DataFrame as a column.
        For instance, if the experiment captured 3 variants and looped over those variants 5 times
        (for a total of 15 images), the first image of each variant will get capture_group 0, the second will get
        capture_group 1, etc. so _get_capture_groups would return pd.Series([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]).
    '''
    num_images = len(variants)
    num_variants = len(set(variants))

    # In case the last capture group isn't complete (due to the experiment ending early)
    # Make sure to round up for the number of capture groups
    num_capture_groups = int(np.ceil(num_images / num_variants))

    full_capture_groups = np.concatenate([
        [i] * num_variants
        for i in range(num_capture_groups)
    ])

    # In case there is a "small capture group" at the end that doesn't have all variants, truncate the list
    capture_groups = full_capture_groups[:num_images]

    return pd.Series(capture_groups)


def _downsample(images_info: pd.DataFrame, downsample_ratio: int) -> pd.DataFrame:
    ''' Take an images_info DataFrame, downsample it based on capture group

    Args:
        images_info: pandas DataFrame with a capture_group column
        downsample_ratio: integer ratio of input rows to output rows: the x in "give me 1 out of every X capture groups"
            If downsample_ratio = 1, keep all images
            If downsample_ratio = 2, keep every other capture group
            If downsample_ratio = 3, keep one in three groups
    Returns:
        a slice of the images_info DataFrame downsampled as instructed
    '''
    return images_info[images_info['capture_group'] % downsample_ratio == 0]


def _filter_to_time_range(
    images_info: pd.DataFrame,
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime]
) -> pd.DataFrame:
    ''' Take an images_info DataFrame and return a version filtered by time range
    If start/end not provided, the appropriate end won't be filtered
    Args:
        images_info: pandas DataFrame with a Timestamp column
        start_time: Optional. Inclusive time value to filter the output data
        end_time: Optional. Inclusive time value to filter the output data
    Returns:
        slice of the input DataFrame matching the start and end values provided
    '''
    # Need to use pd.Timestamp.min here instead of datetime.min because datetime.min is actually earlier than pd's
    # version and causes pd to blow up.
    start_time = start_time or pd.Timestamp.min
    end_time = end_time or pd.Timestamp.max
    return images_info[(images_info['Timestamp'] >= start_time) & (images_info['Timestamp'] <= end_time)]


def sync_from_s3(
    experiment_directory,
    local_sync_directory_path: str,
    downsample_ratio: int = 1,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None,
) -> str:
    ''' Syncs raw images from s3 to a local tmp directory (can optionally be provided)

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_directory_path: The full path of the local parent directory which contains experiment sync
            directories.
        downsample_ratio: Ratio to downsample images by -
            If downsample_ratio = 1, keep all images (default)
            If downsample_ratio = 2, keep half of the capture groups
            If downsample_ratio = 3, keep one in three groups
        start_time: if provided, no images before this datetime will by synced
        end_time: if provided, no images after this datetime will by synced

    Returns:
        Full path of the tmp directory for this experiment
    '''

    local_experiment_dir = os.path.join(local_sync_directory_path, experiment_directory)

    images_info = _get_images_info(experiment_directory)
    downsampled_image_info = _downsample(images_info, downsample_ratio)
    filtered_image_info = _filter_to_time_range(downsampled_image_info, start_time, end_time)
    filenames_to_download = list(filtered_image_info['filename'].values)
    _download_s3_files(experiment_directory, filenames_to_download, local_experiment_dir)

    return local_experiment_dir


def filter_and_reverse_experiment_list(experiment_names, regex):
    # Filter with regex and reverse list of directories to sort most recent first
    # (assumes directory name starts with ISO date)
    return list(reversed([
        experiment_name for experiment_name in experiment_names if re.search(regex, experiment_name)
    ]))


def order_experiment_list_by_isodate_formats(experiment_names):
    experiment_names_with_hyphens_in_isodate = filter_and_reverse_experiment_list(
        experiment_names,
        r'^\d{4}-\d\d-\d\d.'
    )
    experiment_names_without_hyphens_in_isodate = filter_and_reverse_experiment_list(experiment_names, r'^\d{8}.')
    return experiment_names_with_hyphens_in_isodate + experiment_names_without_hyphens_in_isodate


def list_experiments():
    ''' Lists all experiment directories in the "camera-sensor-experiments" bucket

        Returns: a list of experiment names that is filtered and ordered (by isodate formats YYYY-MM-DD & YYYYMMDD)
        The list will be a concatenated set of lists, with the items starting with a list of YYYY-MM-DD formated names
        that are ordered by descending date followed by the same ordering but with a list of YYYYMMDD formatted names.
    '''
    experiment_directories = _list_camera_sensor_experiments_s3_bucket_contents('')

    experiment_names = [directory.rstrip('/') for directory in experiment_directories]

    return order_experiment_list_by_isodate_formats(experiment_names)
