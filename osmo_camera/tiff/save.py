import numpy as np
import tifffile

from ..constants import DNR_TO_TIFF_FACTOR


def _guard_image_fits_in_32_bits(scaled_rgb_image):
    ''' Guard that the values in the scaled image are all within the signed 32-bit range.
    '''
    int32_range = np.iinfo(np.int32)

    if scaled_rgb_image.min() < int32_range.min or scaled_rgb_image.max() > int32_range.max:
        raise ValueError('Pixels in image are out of range')


def as_tiff(rgb_image, output_path):
    ''' Save an RGB Image as a tiff file with signed 32 bit pixel values.

    Args:
        rgb_image: An `RGB image`
        output_path: The full file path (including extension, which must be .tiff) to save the image as

    Returns:
        None
    '''
    scaled_rgb_image = rgb_image * DNR_TO_TIFF_FACTOR

    # Guard that the image will fit in 32-bits. Otherwise, coercing to np.int32 will cause it to warparound
    _guard_image_fits_in_32_bits(scaled_rgb_image)
    rgb_image_as_uint_array = scaled_rgb_image.astype(np.int32)
    tifffile.imsave(output_path, rgb_image_as_uint_array, compress=1)
