import platform
import os
import tempfile
from subprocess import check_call

import boto

''' TODO: delete these notes, fill out functions
Use cases:
* download 1 in every 5 images *by timestamp* (remember there may be multiple variants, we want them all)
* download all images between X and Y timestamp

*note*: I'm kinda trying out type definition syntax in these function declarations:
https://www.python.org/dev/peps/pep-0484/#type-definition-syntax
TODO: I'll either remove the type hints or turn on a checker before I land this branch. I wouldn't add any requirement
    that all functions need to be type hinted.
    In the past, I've found that type hinting in python catches a lot of bugs automatically and removes the need for
    many unit/integration style tests.
    I used it in earlier stages (python 2) when it wasn't as much of a built-in thing.
    Thoughts? Do folks think we should try it out?
'''

### Side effects:
def _list_folder_contents(s3_key: Text) -> list[Text]:
    '''
        tiny function that uses boto to get a list of all of the files in a logical directory off s3.
        Easy to mock out for testing.
    '''


def _get_s3_files(s3_keys: list[Text], output_directory: Text) -> None:
    '''
        downloads specific keys from s3.
        If force=True, deletes the specified files from output_directory before each one is downloaded
        Open questions:
            * I think "s3 sync" parallelizes downloads. Thus if I don't do some kind of parallelizatoin here,
                there will be a significant perf regression.
                Is there a good way to parallelize this that I can borrow/use from boto or maybe aws-cli?
    '''


### Business:
def _get_images_info(experiment_s3_key: Text) -> pd.DataFrame:
    '''
    takes the name of an experiment dir on s3, gets all of the .jpeg file contents.
    Uses _get_folder_contents() for this
    Splits the filenames to get timestamp and variant. converts the timestamps from strings to pd.Timestamp
    uses _get_timestamp_groups to add a capture_group column which starts as 0 for the first image taken of each variant
    and goes up.

    Return a DataFrame with columns: [Timestamp, capture_group, variant, s3_key]
    '''


def _get_capture_groups(images_info: pd.DataFrame) -> pd.Series:
    '''
    gets variants from the images_info DataFrame provided
    returns a column of integers for the logical "capture group" of each image -
    each group will contain one value from each variant and each variant should have one image in each "capture group".

    Tricky part: if there is a missing image from a particular variant, we should skip a value so that the following
    images from that variant are still grouped with values from around the right time interval.
    '''


def _downsample(images_info: pd.DataFrame, downsample_ratio: Optional[int]=1) -> pd.DataFrame:
    '''
    Takes an images_info DataFrame, downsamples it based on capture group

    If downsample_ratio = 1, keep all images
    If downsample_ratio = 2, keep half of the capture groups
    If downsample_ratio = 3, keep one in three groups

    open question: do we need to support fractional downsapling? what would that look like?
    Unless anyone things otherwise I'm gonna go ahead and leave this feature out :)
    '''


def _filter_to_time_range(
    images_info: pd.DataFrame,
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime]
) -> pd.DataFrame:
    ''' Take an images_info DataFrame and return a version filtered by time range
    If start/end not provided, the appropriate end won't be filtered
    '''


def sync_from_s3(
    experiment_directory_name,
    downsample_ratio: Optional[int],
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    local_sync_dir: Optional[Text]=None
):
    ''' Syncs raw images from s3 to a local tmp directory (can optionally be provided)

    Args:
        experiment_dir: The name of the experiment directory in s3
        local_sync_dir: The full path of the directory to sync locally

    Returns:
        Full path of the tmp directory for this experiment
    '''
    # TODO: factor this out into a helper function: _get_sync_directory()
    if not local_sync_dir:
        # On MacOS (Darwin), tempfile.gettempdir() returns a weird auto-generated directory
        # e.g. '/var/folders/nj/269977hs0_96bttwj2gs_jhhp48z54/T'
        # https://stackoverflow.com/questions/847850/cross-platform-way-of-getting-temp-directory-in-python
        local_sync_dir = '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()

    sync_directory_location = os.path.join(local_sync_dir, experiment_directory_name)


    # TODO: the rest of this function will just be calls to the functions stubbed out above
    # Would be better to use boto, but neither boto nor boto3 support sync
    # https://github.com/boto/boto3/issues/358
    command = f'aws s3 sync s3://camera-sensor-experiments/{experiment_directory_name} {sync_directory_location}'
    check_call([command], shell=True)

    return sync_directory_location


def list_experiments():
    ''' Lists all experiment directories in the "camera-sensor-experiments" bucket
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

    # Reverse list of directories to sort most recent first (assumes directory name starts with ISO date)
    return list(reversed(experiment_names))
