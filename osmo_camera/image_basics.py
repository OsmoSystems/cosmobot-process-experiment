
    
def get_channels(image):
    ''' "flatten" the rows and columns of the image so that we have 
    NOTE: channels will come out in whatever order they are stored in the image

    Args:
        image: numpy.ndarray of an openCV-style image
    '''
    rows, cols, num_channels = image.shape
    channels = np.reshape(image, (rows * cols, num_channels)).T
    return channels
    

def crop_image(image, region):
    ''' Crop out a Region of Interest (ROI), returning a new image of just that region

    Args:
        image: numpy.ndarray of an openCV-style image
        region: 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows)
    '''
    start_col, start_row, cols, rows = region
    
    image_subset = image[start_row:start_row+rows, start_col:start_col+cols]
    return image_subset
