# Constant used to convert from 0-1 RGB values to 0-255
MAX_COLOR_VAL = 255


def to_bgr(rgb_image):
    ''' Converts an `RGB image` to a `BGR image`

    Args:
        rgb_image: An `RGB image`

    Returns:
        A `BGR image`
    '''

    # https://www.scivision.co/numpy-image-bgr-to-rgb/
    bgr_image = rgb_image[..., :: -1]

    return bgr_image


def to_int(rgb_image):
    ''' Converts an `RGB image` with 0-1 RGB float values to
        numpy array of 0-255 integer values for use with PIL.

    Args:
        rgb_image: An `RGB image`

    Returns:
        A PIL-compatible 3D numpy array of integer values.
    '''
    return (rgb_image * MAX_COLOR_VAL).astype('uint8')
