import tifffile
from picamraw import PiRawBayer, PiCameraVersion
from ..constants import RAW_BIT_DEPTH, DNR_TO_32_BIT_DEPTH


def as_rgb(raw_image_path):
    ''' Extracts the raw bayer data from a JPEG+RAW file and converts it to an
        `RGB Image` (see definition in README).

    Args:
        raw_image_path: The full path to the JPEG+RAW file

    Returns:
        An `RGB Image`
    '''
    raw_bayer = PiRawBayer(
        filepath=raw_image_path,
        camera_version=PiCameraVersion.V2,
        sensor_mode=0
    )

    # Divide by the bit-depth of the raw data to normalize into the (0,1) range
    rgb_image = raw_bayer.to_rgb() / RAW_BIT_DEPTH

    return rgb_image


def as_uint32_tiff_as_rgb(image_path):
    ''' Opens and reads a 16 bit tiff file and returns
        an `RGB Image` (see definition in README).

    Args:
        image_path: The full path to the tiff file

    Returns:
        An `RGB Image`
    '''
    tiff_image = tifffile.imread(image_path)
    rgb_image = tiff_image / DNR_TO_32_BIT_DEPTH
    return rgb_image
