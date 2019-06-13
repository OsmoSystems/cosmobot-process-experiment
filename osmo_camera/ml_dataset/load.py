import os

import pandas as pd
from tqdm.auto import tqdm

from osmo_camera.s3 import download_s3_files_and_get_local_filepaths


def _get_local_filepaths_for_experiment_df_and_optionally_sync_files(
    rows_for_experiment: pd.DataFrame,
    local_image_files_directory: str,
    sync_files: bool
):
    return download_s3_files_and_get_local_filepaths(
        experiment_directory=rows_for_experiment['experiment'].values[0],
        file_names=rows_for_experiment['image'],
        output_directory_path=local_image_files_directory,
        skip_download=not sync_files
    )


def load_multi_experiment_dataset_csv(dataset_csv_path: str, sync_images: bool = True) -> pd.DataFrame:
    ''' For a pre-prepared ML dataset, load the DataFrame with local image paths, optionally downloading said images
    Note that syncing tends to take a long time.

    Args:
        dataset_csv_path: path to ML dataset CSV. CSV is expected to have at least the following columns:
            'experiment': experiment directory on s3
            'image': image filename on s3
            All other columns are passed through.
        sync_images: whether to sync images. sync_files=True can be slow even if the images have already been
            downloaded, because of the process of checking timestamps against s3 to check for updates.
            Use sync_files=False if you already have the images downloaded.

    Returns:
        DataFrame of the CSV file provided with the additional column 'local_filepath' which will contain file paths of
        the locally stored images.

    Side-effects:
        * if sync_files is True, syncs images corresponding to the ML dataset, from s3 to the standard folder:
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

    if sync_images:
        # Note: this is a very slow progress bar *and* it completes before the process is actually done, but it's still
        # better than nothing considering this is such a long operation.
        print(
            'This is a *very* slow progress bar PLUS it is off by one or something so please wait until I tell you '
            'I am done:'
        )

        local_filepaths = dataset_by_experiment.progress_apply(
            _get_local_filepaths_for_experiment_df_and_optionally_sync_files,
            local_image_files_directory=local_image_files_directory,
            sync_files=True
        )

        print('Done syncing images. thanks for waiting.')
    else:
        local_filepaths = dataset_by_experiment.progress_apply(
            _get_local_filepaths_for_experiment_df_and_optionally_sync_files,
            local_image_files_directory=local_image_files_directory,
            sync_files=False
        )

    full_dataset['local_filepath'] = local_filepaths
    return full_dataset
