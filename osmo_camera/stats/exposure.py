import warnings

COLOR_CHANNELS = 'rgb'
COLOR_CHANNEL_COUNT = len(COLOR_CHANNELS)


def _generate_statistics(rgb_image, overexposed_threshold=0.99, underexposed_threshold=0.1):
    ''' Generate pixel percentage overexposure & underexposure of entire image and overexposure pixel percentage by
        color channel

    Args:
        rgb_image: a `RGB Image`
        overexposed_threshold: threshold at which a color's intensity is overexposed
        underexposed_threshold: threshold at which a color's intensity is underexposed
    Returns:
        dictionary of overexposure & underexposure statistics
    '''

    overexposed_pixel_count_by_channel = (rgb_image > overexposed_threshold).sum(axis=(0, 1))
    underexposed_pixel_count_by_channel = (rgb_image < underexposed_threshold).sum(axis=(0, 1))
    per_channel_pixel_count = rgb_image.size / COLOR_CHANNEL_COUNT

    return {
        'overexposed_threshold': overexposed_threshold,
        'underexposed_threshold': underexposed_threshold,
        'overexposed_percent': overexposed_pixel_count_by_channel.sum() / rgb_image.size,
        'underexposed_percent': underexposed_pixel_count_by_channel.sum() / rgb_image.size,
        ** {
            'overexposed_percent_{}'.format(color):
                overexposed_pixel_count_by_channel[color_index] / per_channel_pixel_count
            for color_index, color in enumerate(COLOR_CHANNELS)
        },
        ** {
            'underexposed_percent_{}'.format(color):
                underexposed_pixel_count_by_channel[color_index] / per_channel_pixel_count
            for color_index, color in enumerate(COLOR_CHANNELS)
        }
    }


def warn_if_exposure_out_of_range(rgb_images, overexposed_percent_threshold=0.01, underexposed_percent_threshold=0.01):
    for raw_image_path, rgb_image in rgb_images.items():
        stats = _generate_statistics(rgb_image)

        if stats["overexposed_percent"] > overexposed_percent_threshold:
            warnings.warn("raw_image_path - overexposed percent")

        if stats["underexposed_percent"] > underexposed_percent_threshold:
            warnings.warn("raw_image_path - underexposed percent")
