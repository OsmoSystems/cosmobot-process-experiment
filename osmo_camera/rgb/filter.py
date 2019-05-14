import logging
import re
from textwrap import dedent

logger = logging.getLogger('osmo_camera.rgb.filter')


def select_channels(image, channels):
    ''' Selectively filters channels on an RGB image

    Args:
        image: An RGB image
        channels: String specifying channels to preserve in output image.
            Ex. 'GB' to keep Green and Blue channels and set all Red pixels to 0.
    Returns:
        An RGB image with only the selected channels.
    '''
    channels = channels.lower()
    if 'r' not in channels:
        image[:, :, 0] = 0
    if 'g' not in channels:
        image[:, :, 1] = 0
    if 'b' not in channels:
        image[:, :, 2] = 0
    extra_args = re.sub(r'[rgb\s]', '', channels)
    if len(extra_args) > 0:
        logger.warning(dedent(
            f'''\
            Ignoring unrecognized channel argument "{extra_args}" provided to select_channels.
            '''
        ))
    return image
