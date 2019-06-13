import os

import pandas as pd
from tqdm.auto import tqdm

from osmo_camera.s3 import naive_s3_sync


def _get_files_for_experiment_df(experiment_df, local_image_files_directory):
    # All rows in the group are the same experiment, so just grab the first one
    experiment_directory = experiment_df['experiment'].values[0]
    return naive_s3_sync(
        experiment_directory=experiment_directory,
        file_names=experiment_df['image'],
        output_directory_path=local_image_files_directory,
    )


def load_multi_experiment_dataset_csv(dataset_csv_path: str) -> pd.DataFrame:
    ''' For a pre-prepared ML dataset, load the DataFrame with local image paths, optionally downloading said images
    Note that syncing tends to take a long time, though syncing for individual experiments will be skipped if all files
    are already downloaded.

    Args:
        dataset_csv_path: path to ML dataset CSV. CSV is expected to have at least the following columns:
            'experiment': experiment directory on s3
            'image': image filename on s3
            All other columns are passed through.

    Returns:
        DataFrame of the CSV file provided with the additional column 'local_filepath' which will contain file paths of
        the locally stored images.

    Side-effects:
        * syncs images corresponding to the ML dataset, from s3 to the standard folder:
            ~/osmo/cosmobot-data-sets/{CSV file name without extension}/
        * prints status messages so that the user can keep track of this very slow operation
        * calls tqdm.auto.tqdm.pandas() which patches pandas datatypes to have `.progress_apply()` methods
    '''
    # Side effect: patch pandas datatypes to have .progress_apply() methods
    tqdm.pandas()

    full_dataset = pd.read_csv(dataset_csv_path)

    dataset_csv_filename = os.path.basename(dataset_csv_path)
    local_image_files_directory = os.path.join(
        os.path.expanduser('~/osmo/cosmobot-data-sets/'),
        os.path.splitext(dataset_csv_filename)[0]  # Get rid of the .csv part
    )

    dataset_by_experiment = full_dataset.groupby('experiment', as_index=False, group_keys=False)

    # Note: this is a very slow progress bar *and* it completes before the process is actually done, but it's still
    # better than nothing considering this is such a long operation.
    print(
        'This can be a *very* slow, uneven progress bar PLUS it is off by one or something so please wait until I tell '
        'you I am done:'
    )

    local_filepaths = dataset_by_experiment.progress_apply(
        _get_files_for_experiment_df,
        local_image_files_directory=local_image_files_directory
    )

    print('Done syncing images. thanks for waiting.')

    full_dataset['local_filepath'] = local_filepaths
    return full_dataset
