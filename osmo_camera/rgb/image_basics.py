# TODO: name this file something different? Maybe break these functions out?
import numpy as np


# TODO: make sure this works
def get_channels(image):
    ''' Convert an image into multiple grayscale images, one for each channel

    Args:
        image: numpy.ndarray of an openCV-style image
    Returns:
        np.array of arrays, where each sub-array is a channel from the original image
        NOTE: channels will come out in whatever order they are stored in the image
    '''
    rows, cols, num_channels = image.shape
    channels = np.reshape(image, (rows * cols, num_channels)).T
    return channels


def crop_image(image, ROI_definition):
    ''' Crop out a Region of Interest (ROI), returning a new image of just that ROI

    Args:
        image: numpy.ndarray image
        ROI_definition: 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
    Returns:
        numpy.ndarray image containing pixel values from the input image
    '''
    start_col, start_row, cols, rows = ROI_definition

    image_crop = image[start_row:start_row+rows, start_col:start_col+cols]
    return image_crop
