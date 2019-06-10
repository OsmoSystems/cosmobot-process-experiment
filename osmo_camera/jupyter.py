import os

import ipywidgets
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tqdm.auto import tqdm

from osmo_camera.s3 import list_experiments, get_local_filepaths


def select_experiment():
    ''' Display a dropdown of experiment names, pulled from s3
    '''

    selection = ipywidgets.Dropdown(options=list_experiments(), value=None, layout={'width': 'initial'})
    print('Select experiment to process:')
    display(selection)
    return selection


def show_image(rgb_image, figsize=None, title=''):
    ''' Show an image in an ipython notebook.

    Args:
        rgb_image: numpy.ndarray of an RGB image
        figsize: 2-tuple of desired figure size in inches; will be passed to `plt.figure()`
    '''
    plt.figure(figsize=figsize)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.show()


def _make_solid_color_image(cv_color):
    image = np.zeros((10, 10, len(cv_color)), np.uint8)
    image[:] = cv_color
    return image


def show_color(cv_color):
    image_size = 0.5  # just print a little swatch. This will be interpreted in inches
    show_image(_make_solid_color_image(cv_color), figsize=(image_size, image_size))


def _get_flattened_channels(image):
    ''' Convert an image into linear arrays for each channel

    Args:
        image: numpy.ndarray of an openCV-style image
    Returns:
        np.array of arrays, where each sub-array is a channel from the original image
        NOTE: channels will come out in whatever order they are stored in the image
    '''
    rows, cols, num_channels = image.shape
    channels = np.reshape(image, (rows * cols, num_channels)).T
    return channels


def plot_histogram(image, title='', bins=1024):
    ''' Plot a histogram of the image

    Args:
        image: numpy array of an RGB image
        title: title of the data set, will be used in the plot title if minimal=False
        bins: number of bins for the histogram - it's recommended to use the intensity resolution of your image.
            Defaults to 2 ** 10 = 1024
    Returns:
        plotly FigureWidget. Call display() on this to view it.
    '''
    # Assume this is one of our standard RGB images with values between 0 and 1
    range_per_channel = (0, 1)
    red, green, blue = _get_flattened_channels(image)

    histograms_and_bin_edges_by_color = {
        color_name: np.histogram(channel, bins, range=range_per_channel, density=True)
        for color_name, channel
        in {'red': red, 'green': green, 'blue': blue}.items()
    }

    traces = [
        go.Scatter(
            x=bin_edges,
            y=histogram,
            name=color,
            mode='lines',
            line={
                'color': color,
                'width': 1,
            },
            fill='tozeroy',
        )
        for color, (histogram, bin_edges)
        in histograms_and_bin_edges_by_color.items()
    ]

    layout_kwargs = {
        'title': f'{title} pixel density histogram',
        'xaxis': {'title': 'Channel value'},
        'yaxis': {'title': 'Density'}
    }
    layout = go.Layout(**layout_kwargs)

    return go.FigureWidget(data=traces, layout=layout)


def load_multi_experiment_dataset_csv(dataset_csv_path: str, sync_images: bool = True) -> pd.DataFrame:
    ''' For a pre-prepared ML dataset, load the DataFrame with local image paths, optionally downloading said images
    Note that syncing tends to take a long time.

    Args:
        dataset_csv_path: path to ML dataset CSV. CSV is expected to have at least the following columns:
            'experiment': experiment directory on s3
            'image': image filename on s3
        sync_images: whether to sync images. sync_images=True can be slow even if the images have already been
            downloaded, because of the process of checking timestamps against s3 to check for updates.
            Use sync_images=False if you already have the images downloaded.

    Returns:
        DataFrame of the CSV file provided with the additional column 'local_filepath' which will contain file paths of
        the locally stored images.

    Side-effects:
        * if sync_images is True, syncs images corresponding to the ML dataset to the standard folder:
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

    if sync_images:
        # Note: this is a very slow progress bar *and* it completes before the process is actually done, but it's still
        # better than nothing considering this is such a long operation.
        print(
            'This is a *very* slow progress bar PLUS it is off by one or something so please wait until I tell you '
            'I am done:'
        )

    # Use a groupby to get image file paths one experiment at a time
    local_filepaths = full_dataset.groupby('experiment', as_index=False).progress_apply(
        lambda rows_for_experiment: get_local_filepaths(
            experiment_directory=rows_for_experiment['experiment'].values[0],
            file_names=rows_for_experiment['image'],
            output_directory_path=local_image_files_directory,
            sync_images=sync_images
        ),
    )

    if sync_images:
        print('Done syncing images. thanks for waiting.')

    full_dataset['local_filepath'] = local_filepaths
    return full_dataset
