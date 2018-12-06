import numpy as np

# Default trimmedness: discard anything more than 4 standard deviations from a central value
DEFAULT_TRIM_STD = 4


def _trim_data_to_stdev(sample, trim_std):
    ''' Trim the farther reaches of a data set based on a central value and standard deviation

    Arguments:
        sample: 1-dimensional numpy array to be trimmed
        trim_std: number of standard deviations away from the median to keep
            e.g. if 0, no data will be kept. If 2, anything within 2 standard deviations of the median will be kept
    Return:
        trimmed version of the sample with anything outside of `trim_std` standard deviations of the mean removed
    '''

    median = np.median(sample)
    stdev = np.std(sample)

    allowed_half_width = stdev * trim_std

    min_ = median - allowed_half_width
    max_ = median + allowed_half_width

    trimmed_sample = sample[(sample > min_) & (sample < max_)]

    return trimmed_sample


def median_seeded_outlier_removed_mean(sample, trim_std=DEFAULT_TRIM_STD):
    ''' Calculate the Median-Seeded, Outlier-Removed Mean (~MSORM~) of a flat data sample

    Arguments:
        sample: 1-dimensional numpy array to find the central value of
        trim_std: number of standard deviations away from the median to keep
    Return:
        the mean of the sample after outliers have been removed.
        Outliers are removed based on their distance from the median
    '''
    if len(sample.shape) > 1:
        raise ValueError(
            'median_seeded_outlier_removed_mean() only supports flat arrays. Sample has shape {}.'.format(sample)
        )
    if np.all(sample == sample[0]):
        # If there is no variation in the sample, just return any value
        return sample[0]

    trimmed_sample = _trim_data_to_stdev(
        sample,
        trim_std
    )
    return np.mean(trimmed_sample)


msorm = median_seeded_outlier_removed_mean

msorm(np.array([-30, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 55]))  # 5.06 or something


def image_msorm(image, trim_std=DEFAULT_TRIM_STD):
    ''' Calculate the Median-Seeded, Outlier-Removed Mean (~MSORM~ for short) of an RGB image

    Arguments:
        image: RGB image numpy array to find the central value of
        trim_std: number of standard deviations away from the median to keep
    Return:
        1D numpy array: for each channel,
            the mean of the sample after outliers have been removed.
            Outliers are removed based on their distance from the median
    '''
    channel_indices = range(image.shape[2])
    flattened_channels = [
        image[:, :, channel].flatten()
        for channel in channel_indices
    ]
    return np.array([
        msorm(channel, trim_std)
        for channel in flattened_channels
    ])


rgb_image = np.array([
    [[1, 2, 3],  [1, 2, 3]],
    [[1, 2, 3],  [1, 2, 3]],
])
image_msorm(rgb_image)  # np.array([1,2,3])


def image_stack_msorm(image_stack, trim_std=DEFAULT_TRIM_STD):
    ''' Calculate the Median-Seeded, Outlier-Removed Mean (~MSORM~ for short) of a "stack" of RGB images

    Arguments:
        image_stack: RGB image "stack" numpy array to find the central value of.
            This is a 4-dimensional numpy array where the first dimension iterates over images.
        trim_std: number of standard deviations away from the median to keep
    Return:
        1D numpy array: for each channel,
            the mean of the sample after outliers have been removed.
            Outliers are removed based on their distance from the median
    '''
    channel_indices = range(image_stack.shape[3])
    flattened_channels = [
        image_stack[:, :, :, channel].flatten()
        for channel in channel_indices
    ]
    return np.array([
        msorm(channel, trim_std)
        for channel in flattened_channels
    ])


rgb_image_stack = np.array([
    rgb_image, rgb_image
])
image_stack_msorm(rgb_image_stack)  # np.array([1,2,3])
