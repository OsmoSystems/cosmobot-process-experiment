import numpy as np
import tifffile

from ..constants import DNR_TO_DN_BIT_DEPTH


def as_tiff(rgb_image, output_path):
    ''' Save an RGB Image as a tiff file with signed 32 bit pixel values.

    Args:
        rgb_image: An `RGB image`
        output_path: The full file path (including extension, which must be .tiff) to save the image as

    Returns:
        None
    '''
    rgb_image_as_uint_array = (rgb_image * (DNR_TO_DN_BIT_DEPTH)).astype(np.int32)
    tifffile.imsave(output_path, rgb_image_as_uint_array, compress=1)
