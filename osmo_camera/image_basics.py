import numpy as np


def get_channels(image):
    ''' "flatten" the rows and columns of the image so that we have

    Args:
        image: numpy.ndarray of an openCV-style image
    Returns:
        np.array of arrays, where each sub-array is a channel from the original image
        NOTE: channels will come out in whatever order they are stored in the image
    '''
    rows, cols, num_channels = image.shape
    channels = np.reshape(image, (rows * cols, num_channels)).T
    return channels


def crop_image(image, region):
    ''' Crop out a Region of Interest (ROI), returning a new image of just that region

    Args:
        image: numpy.ndarray of an openCV-style image
        region: 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
    Returns:
        numpy.ndarray of an openCV-style image containing pixel values from the input image
    '''
    start_col, start_row, cols, rows = region

    image_subset = image[start_row:start_row+rows, start_col:start_col+cols]
    return image_subset
