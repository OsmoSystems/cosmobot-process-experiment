import logging
from textwrap import dedent

from PIL import Image

logger = logging.getLogger('osmo_camera.rgb.convert')

# Constant used to convert from 0-1 RGB values to 0-255
MAX_COLOR_VALUE = 255


def to_bgr(rgb_image):
    ''' Converts an `RGB image` to a `BGR image`

    Args:
        rgb_image: An `RGB image`

    Returns:
        A `BGR image`
    '''

    # https://www.scivision.co/numpy-image-bgr-to-rgb/
    bgr_image = rgb_image[..., :: -1]

    return bgr_image


def to_PIL(rgb_image):
    ''' Converts an `RGB image` with 0-1 RGB float values to PIL image object.

    Args:
        rgb_image: An `RGB image`

    Returns:
        A PIL image object.
    '''
    # Count the number of items which exceed the expected threshold
    count_above_threshold = (rgb_image > 1).sum()
    if count_above_threshold > 0:
        logger.warning(dedent(
            f'''\
            Found {count_above_threshold} items exceeded maxmimum expected value of 1.
            These values will be truncated to the maximum output value of {MAX_COLOR_VALUE}.
            in the converted image\
            '''
        ))
        rgb_image[rgb_image > 1] = 1
    return Image.fromarray((rgb_image * MAX_COLOR_VALUE).astype('uint8'))
