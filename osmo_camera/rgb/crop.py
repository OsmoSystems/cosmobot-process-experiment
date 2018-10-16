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
