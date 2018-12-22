import numpy as np
import tifffile

from ..constants import DNR_TO_16_BIT_DEPTH


def as_uint16_tiff(rgb_image, output_path):
    ''' Save an RGB Image as a tiff file.

    Args:
        rgb_image: An `RGB image`
        output_path: The full file path (including extension, which must be .tiff) to save the image as

    Returns:
        None
    '''
    rgb_image_as_uint_array = (rgb_image * (DNR_TO_16_BIT_DEPTH)).astype(np.uint16)
    tifffile.imsave(output_path, rgb_image_as_uint_array, compress=1)
