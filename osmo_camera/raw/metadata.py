from collections import namedtuple
from datetime import datetime

import PIL.Image
import PIL.ExifTags


# Just the EXIF tags we care about
ExifTags = namedtuple("ExifTags", ["capture_datetime", "iso", "exposure_time"])


def _read_exif_tags(image_path):
    """
    Uses (an "experimental" private function from) PIL to read EXIF tags from an image file.
    Returns a dictionary of tag names to values
    """
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
    date_time_string = tags["DateTimeOriginal"]

    # For DateTimeOriginal, PIL _getexif returns an ISO8601-ish string
    return datetime.strptime(date_time_string, "%Y:%m:%d %H:%M:%S")


def _parse_iso(tags):
    # For ISOSpeedRatings PIL _getexif returns an int
    return tags["ISOSpeedRatings"]


def _parse_exposure_time(tags):
    exposure = tags["ExposureTime"]

    # For ExposureTime, PIL _getexif returns a tuple of (numerator, denominator)
    numerator, denominator = exposure
    return numerator / denominator


def get_exif_tags(raw_image_path):
    """ Extracts relevant EXIF tags from a JPEG+RAW file.

    Args:
        raw_image_path: The full file path of the JPEG+RAW file to extract metadata from

    Returns:
        Relevant EXIF tags, as an ExifTags namedtuple
    """
    tags = _read_exif_tags(raw_image_path)

    return ExifTags(
        capture_datetime=_parse_date_time_original(tags),
        iso=_parse_iso(tags),
        exposure_time=_parse_exposure_time(tags),
    )
