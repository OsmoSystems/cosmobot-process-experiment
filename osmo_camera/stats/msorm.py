import numpy as np

from ..constants import COLOR_CHANNEL_INDICES


# Default trimmedness: discard anything more than 4 standard deviations from a central value
DEFAULT_TRIM_STDEV = 4


def _trim_data_to_stdev(sample, trim_stdev):
    ''' Trim the farther reaches of a data set based on a central value and standard deviation

    Arguments:
        sample: 1-dimensional numpy array to be trimmed
        trim_stdev: number of standard deviations away from the median to keep
            e.g. if 0, only values matching the median will be kept.
                 If 2, anything within 2 standard deviations of the median will be kept
    Return:
        trimmed version of the sample with anything outside of `trim_stdev` standard deviations of the mean removed
    '''

    median = np.median(sample)
    stdev = np.std(sample)

    allowed_half_width = stdev * trim_stdev

    min_ = median - allowed_half_width
    max_ = median + allowed_half_width

    trimmed_sample = sample[(sample >= min_) & (sample <= max_)]

    return trimmed_sample


def median_seeded_outlier_removed_mean(sample, trim_stdev=DEFAULT_TRIM_STDEV):
    ''' Calculate the Median-Seeded, Outlier-Removed Mean (~MSORM~) of a flat data sample

    Arguments:
        sample: 1-dimensional numpy array to find the central value of
        trim_stdev: number of standard deviations away from the median to keep
    Return:
        the mean of the sample after outliers have been removed.
        Outliers are removed based on their distance from the median
    '''
    if len(sample.shape) != 1:
        raise ValueError(
            f'median_seeded_outlier_removed_mean() only supports flat arrays. Sample has shape {sample}.'
        )

    trimmed_sample = _trim_data_to_stdev(
        sample,
        trim_stdev
    )
    return np.mean(trimmed_sample)


msorm = median_seeded_outlier_removed_mean


def _validate_rgb_image_shape(rgb_image, image_name):
    shape = rgb_image.shape
    if len(shape) != 3:
        raise ValueError(f'{image_name} is expected to have 3 dimensions but had shape {rgb_image.shape}')

    num_color_channels = rgb_image.shape[2]
    expected_num_color_channels = len(COLOR_CHANNEL_INDICES)
    if num_color_channels != expected_num_color_channels:
        raise ValueError(
            f'{image_name} is expected to have {expected_num_color_channels} '
            f'channels but had {num_color_channels}. (shape={rgb_image.shape})'
        )


def image_msorm(image, trim_stdev=DEFAULT_TRIM_STDEV):
    ''' Calculate the Median-Seeded, Outlier-Removed Mean (~MSORM~ for short) of an RGB image

    Arguments:
        image: RGB image numpy array to find the central value of
        trim_stdev: number of standard deviations away from the median to keep
    Return:
        1D numpy array: for each channel, the mean of the sample after outliers have been removed.
            Outliers are removed based on their distance from the median
    '''
    _validate_rgb_image_shape(image, 'Image passed to image_msorm()')

    flattened_channels = [
        image[:, :, channel].flatten()
        for channel in COLOR_CHANNEL_INDICES
    ]
    return np.array([
        msorm(channel, trim_stdev)
        for channel in flattened_channels
    ])


def image_stack_msorm(image_stack, trim_stdev=DEFAULT_TRIM_STDEV):
    ''' Calculate the Median-Seeded, Outlier-Removed Mean (~MSORM~ for short) of a "stack" of RGB images

    Arguments:
        image_stack: RGB image "stack" numpy array to find the central value of.
            This is a 4-dimensional numpy array where the first dimension iterates over images.
        trim_stdev: number of standard deviations away from the median to keep
    Return:
        1D numpy array: for each channel,
            the mean of the sample (across all images) after outliers have been removed.
            Outliers are removed based on their distance from the median
    '''
    _validate_rgb_image_shape(image_stack[0], 'First image in stack passed to image_stack_msorm()')

    flattened_channels = [
        image_stack[:, :, :, channel].flatten()
        for channel in COLOR_CHANNEL_INDICES
    ]
    return np.array([
        msorm(channel, trim_stdev)
        for channel in flattened_channels
    ])
