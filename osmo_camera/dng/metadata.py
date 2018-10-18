from collections import namedtuple
from datetime import datetime
import os

import PIL.Image
import PIL.ExifTags


# Just the EXIF tags we care about
ExifTags = namedtuple(
    'ExifTags',
    [
        'capture_datetime',
        'iso',
        'exposure_time'
    ]
)


def _read_exif_tags(image_path):
    '''
    Uses (an "experimental" private function from) PIL to read EXIF tags from an image file.
    Returns a dictionary of tag names to values
    '''
    PIL_image = PIL.Image.open(image_path)
    EXIF_CODES_TO_NAMES = PIL.ExifTags.TAGS

    # _getexif() returns a dictionary of {tag code: tag value}. Use PIL.ExifTags.TAGS dictionary of {tag code: tag name}
    # to construct a more digestible dictionary of {tag name: tag value}
    tags = {
        EXIF_CODES_TO_NAMES[tag_code]: tag_value
        for tag_code, tag_value in PIL_image._getexif().items()
        if tag_code in EXIF_CODES_TO_NAMES
    }

    return tags


def _parse_date_time_original(tags):
    date_time_string = tags['DateTimeOriginal']

    # For DateTimeOriginal, PIL _getexif returns an ISO8601-ish string
    return datetime.strptime(date_time_string, '%Y:%m:%d %H:%M:%S')


def _parse_iso(tags):
    # For ISOSpeedRatings PIL _getexif returns an int
    return tags['ISOSpeedRatings']


def _parse_exposure_time(tags):
    exposure = tags['ExposureTime']

    # For ExposureTime, PIL _getexif returns a tuple of (numerator, denominator)
    numerator, denominator = exposure
    return numerator / denominator


def get_exif_tags(dng_image_path):
    ''' Extracts relevant EXIF tags for a .DNG file.
    Assumes that a .JPEG file of the same name, which actually contains the EXIF data, lives alongside

    Args:
        dng_image_path: The full file path of the .DNG file to get metadata for

    Returns:
        Relevant EXIF tags, as an ExifTags namedtuple
    '''
    dng_image_path_root, dng_image_extension = os.path.splitext(dng_image_path)
    raw_image_path = f'{dng_image_path_root}.jpeg'

    tags = _read_exif_tags(raw_image_path)

    return ExifTags(
        capture_datetime=_parse_date_time_original(tags),
        iso=_parse_iso(tags),
        exposure_time=_parse_exposure_time(tags),
    )
