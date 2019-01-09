import numpy as np
import tifffile

from ..constants import DNR_TO_TIFF_FACTOR


class DataTruncationError(ValueError):
    pass


def _guard_rgb_image_fits_in_padded_range(rgb_image):
    ''' Guard that the values in the rgb image are all within the padding left over
    after dividing the signed 32-bit range by the DNR_TO_TIFF_FACTOR.
    '''
    int32_range = np.iinfo(np.int32)
    allowed_min = int32_range.min / DNR_TO_TIFF_FACTOR
    allowed_max = int32_range.max / DNR_TO_TIFF_FACTOR

    if rgb_image.min() < allowed_min or rgb_image.max() > allowed_max:
        raise DataTruncationError(
            f'Pixels in image are out of range. '
            f'Image range: [{rgb_image.min()}, {rgb_image.max()}]. '
            f'Allowed range: [{allowed_min}, {allowed_max}]'
        )


def as_tiff(rgb_image, output_path):
    ''' Save an RGB Image as a tiff file with signed 32 bit pixel values.

    Args:
        rgb_image: An `RGB image`
        output_path: The full file path (including extension, which must be .tiff) to save the image as

    Returns:
        None
    '''
    # Guard that the image will fit in 32 bits, after applying the DNR_TO_TIFF_FACTOR.
    # Otherwise, coercing to np.int32 will cause it to wrap around
    _guard_rgb_image_fits_in_padded_range(rgb_image)

    scaled_rgb_image = rgb_image * DNR_TO_TIFF_FACTOR
    rgb_image_as_uint_array = scaled_rgb_image.astype(np.int32)
    tifffile.imsave(output_path, rgb_image_as_uint_array, compress=1)
