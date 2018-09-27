import numpy as np
import cv2


def get_channel_minimums(image):
    ''' Minimum value for each channel in an image.
    '''
    return np.amin(image, axis=(0, 1))


def get_channel_means(image):
    ''' Mean value for each channel in an image.
    '''
    # Cv2.mean includes an extra zero for some reason
    return cv2.mean(image)[:-1]


def get_channel_maximums(image):
    ''' Maximum value for each color component in an image.
    '''
    return np.amax(image, axis=(0, 1))


def get_channel_stdevs(image):
    mean, stdev = cv2.meanStdDev(image)
    # For some reason, stdev comes out as a 2d array; flatten it
    return np.reshape(stdev, 3)


def get_channel_medians(image):
    return np.median(image, axis=(0, 1))
