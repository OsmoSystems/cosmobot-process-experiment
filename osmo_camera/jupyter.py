from matplotlib import pyplot as plt
import numpy as np
from plotly.offline import iplot
import plotly.graph_objs as go

from .rgb.image_basics import get_channels


def _make_solid_color_image(cv_color):
    image = np.zeros((10, 10, len(cv_color)), np.uint8)
    image[:] = cv_color
    return image


# TODO: should this allow passing through any matplotlib figure kwargs?
def show_image(rgb_image, figsize=None):
    ''' Show an image in an ipython notebook.

    Args:
        rgb_image: numpy.ndarray of an RGB image
        figsize: 2-tuple of desired figure size in inches; will be passed to `plt.figure()`
    '''
    plt.figure(figsize=figsize)
    plt.imshow(rgb_image)
    plt.show()


def show_color(cv_color):
    image_size = 0.5  # just print a little swatch. This will be interpreted in inches
    show_image(_make_solid_color_image(cv_color), figsize=(image_size, image_size))


def plot_histogram(image, minimal=True):
    ''' Plot a histogram of the image
    '''
    # TODO: debug this?
    # assumes image is in "green, blue, red" (openCV default) channel format
    blue, green, red = get_channels(image)

    max_value = np.iinfo(green.dtype).max
    bins = max_value

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
            mode='line',
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

    figure = go.Figure(data=traces, layout=layout)

    iplot(figure, show_link=False)
