import ipywidgets
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go

from osmo_camera.s3 import list_experiments


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
