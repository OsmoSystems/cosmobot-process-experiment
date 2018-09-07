import cv2
from matplotlib import pyplot as plt
import numpy as np
from plotly.offline import iplot
import plotly.graph_objs as go

from .image_basics import get_channels


def choose_regions(image):
    ''' Funky interaction to select regions within an image.
    READ THIS:
    When you call this, the user must:
    1. go to the window that pops up
    2. click + drag to select a region
    3. PRESS ENTER ONCE. Pressing enter multiple times will save the same region again
    4. return to step 2 until you've selected all the regions you want
    5. after pressing enter the last time, close the window by pressing Esc a couple of times.

    Arguments:
        image: numpy.ndarray of an openCV-style image
    Returns:
        numpy 2d array, essentially an iterable containing iterables of (start_col, start_row, cols, rows)
        corresponding to the regions that you selected.
    '''
    window_name = 'ROIs selection'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)  # WINDOW_GUI_EXPANDED seems to allow you to resize the window
    
    # Resize the window to a manageable default.
    window_size = 600  # in pixels
    cv2.resizeWindow(window_name, window_size, window_size)

    regions = cv2.selectROIs(window_name, image)
    cv2.waitKey()

    cv2.destroyWindow(window_name)
    return regions


def _make_solid_color_image(cv_color):
    image = np.zeros((10, 10, len(cv_color)), np.uint8)
    image[:] = cv_color
    return image


def show_image(image, figsize=None):
    ''' Show an image in an ipython notebook.

    Args:
        image: numpy.ndarray of an openCV-style image
        figsize: 2-tuple of desired figure size in inches; will be passed to `plt.figure()`
    '''
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def show_color(cv_color):
    image_size = 0.5  # just print a little swatch. This will be interpreted in inches
    show_image(_make_solid_color_image(cv_color), figsize=(image_size, image_size))


def plot_histogram(image, minimal=True):
    ''' Plot a histogram of the image
    '''
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
