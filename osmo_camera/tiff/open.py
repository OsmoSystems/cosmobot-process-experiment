import tifffile
from ..constants import DNR_TO_TIFF_FACTOR


def as_rgb(image_path):
    ''' Opens and reads a 32 bit tiff file and returns
        an `RGB Image` (see definition in README).

    Args:
        image_path: The full path to the tiff file

    Returns:
        An `RGB Image`
    '''
    tiff_image = tifffile.imread(image_path)
    rgb_image = tiff_image / DNR_TO_TIFF_FACTOR
    return rgb_image
