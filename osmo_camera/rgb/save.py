import numpy as np
from PIL import Image
import tifffile

from osmo_camera.constants import DNR_TO_16_BIT_DEPTH


def as_file(rgb_image, output_path):
    ''' Save an RGB Image as a file. Note that this downsamples to an 8-bit image. Only tested with .png files

    Args:
        rgb_image: An `RGB image`
        output_path: The full file path (including extension) to save the image as

    Returns:
        None
    '''
    # PIL Image expects a unsigned int 8-bit image
    rgb_image_as_uint_array = (rgb_image * (2 ** 8 - 1)).astype(np.uint8)
    img = Image.fromarray(rgb_image_as_uint_array, 'RGB')
    img.save(output_path)


def as_uint16_tiff(rgb_image, output_path):
    ''' Save an RGB Image as a tiff file.

    Args:
        rgb_image: An `RGB image`
        output_path: The full file path (including extension, which must be .tiff) to save the image as

    Returns:
        None
    '''
    rgb_image_as_uint_array = (rgb_image * (DNR_TO_16_BIT_DEPTH)).astype(np.uint16)
    tifffile.imsave(output_path, rgb_image_as_uint_array)
