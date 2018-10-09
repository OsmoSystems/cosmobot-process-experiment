import ipywidgets
from IPython.display import display

from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go

from osmo_camera.rgb.image_basics import get_channels
from osmo_camera.s3 import list_experiments
from osmo_camera.select_ROI import draw_ROIs_on_image


def select_experiment():
    ''' Display a dropdown of experiment names, pulled from s3
    '''

    selection = ipywidgets.Dropdown(options=list_experiments(), value=None, layout={'width': 'initial'})
    print('Select experiment to process:')
    display(selection)
    return selection


def show_image(rgb_image, figsize=None, title='', ROI_definitions=None):
    ''' Show an image in an ipython notebook.

    Args:
        rgb_image: numpy.ndarray of an RGB image
        figsize: 2-tuple of desired figure size in inches; will be passed to `plt.figure()`
        ROI_definitions: Optional. If provided, will draw the ROIs on the displayed image
    '''
    plt.figure(figsize=figsize)

    if ROI_definitions:
        rgb_image_with_ROIs = draw_ROIs_on_image(rgb_image, ROI_definitions)
        plt.imshow(rgb_image_with_ROIs)
    else:
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


def plot_histogram(image, minimal=True):
    ''' Plot a histogram of the image
    '''
    # Assume this is one of our standard RGB images with values between 0 and 1
    red, green, blue = get_channels(image)
    max_value = 1
    bins = 200

    histograms_and_bin_edges_by_color = {
        color_name: np.histogram(channel, bins, range=(0, max_value), density=True)
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
        'height': 300,
        'showlegend': False,
    } if minimal else {
        'title': 'Pixel density histogram',
        'xaxis': {'title': 'Channel value'},
        'yaxis': {'title': 'Density'}
    }
    layout = go.Layout(**layout_kwargs)

    return go.FigureWidget(data=traces, layout=layout)
