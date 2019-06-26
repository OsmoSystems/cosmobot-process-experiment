import numpy as np


def select_channels(rgb_image, channels):
    """ Selectively filters channels on an RGB image

    Args:
        image: An RGB image
        channels: Lowercase string specifying channels to preserve in output image.
            Ex. 'gb' to keep green and blue channels and set all red pixels to 0.
    Returns:
        An RGB image with only the selected channels.
    """
    colors = "rgb"
    if set(channels) - set(colors):
        raise ValueError(f'Unexpected channel value in select_channels: "{channels}"')

    channels_to_drop = set(colors) - set(channels)
    channel_indices_to_drop = [colors.index(channel) for channel in channels_to_drop]

    image = np.copy(rgb_image)
    for index in channel_indices_to_drop:
        image[:, :, index] = 0
    return image
