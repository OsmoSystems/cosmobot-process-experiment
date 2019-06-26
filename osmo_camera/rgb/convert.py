import logging
from textwrap import dedent

import numpy as np
from PIL import Image

logger = logging.getLogger("osmo_camera.rgb.convert")

# Constant used to convert from 0-1 RGB values to 0-255
MAX_COLOR_VALUE = 255


def to_bgr(rgb_image):
    """ Converts an `RGB image` to a `BGR image`

    Args:
        rgb_image: An `RGB image`

    Returns:
        A `BGR image`
    """

    # https://www.scivision.co/numpy-image-bgr-to-rgb/
    bgr_image = rgb_image[..., ::-1]

    return bgr_image


def to_PIL(rgb_image):
    """ Converts an `RGB image` with 0-1 RGB float values to PIL image object.

    Args:
        rgb_image: An `RGB image`

    Returns:
        A PIL image object.
    """
    # Count the number of items which will not convert nicely to uint8 and will be truncated
    count_out_of_range = (rgb_image > 1).sum() + (rgb_image < 0).sum()
    if count_out_of_range > 0:
        logger.warning(
            dedent(
                f"""\
                Found {count_out_of_range} items outside acceptable value range of 0-1.
                Values greater than 1 will be truncated to the maximum output value of {MAX_COLOR_VALUE}
                in the converted image.
                Values less than 0 will be truncated to 0 in the converted image.\
                """
            )
        )
        rgb_image = np.clip(rgb_image, 0, 1)
    return Image.fromarray((rgb_image * MAX_COLOR_VALUE).astype("uint8"))
