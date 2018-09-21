import datetime
import os

import exifread


EXIF_DATE_TIME_ORIGINAL_TAG = 'EXIF DateTimeOriginal'


# TODO: stop assuming the jpeg version of the file is alongside the DNG file. Either copy over metadata to .DNG, or..?
def capture_date(dng_image_path):
    ''' Extracts original capture date from a .DNG file

    Captue date is stored as an EXIF key formatted like:
        'EXIF DateTimeOriginal': (0x0132) ASCII=2018:09:10 20:01:19 @ 59140

    the right side is an EXIF key value; getting ex_key.values gives you a nice ISO8601-ish string

    Args:
        dng_image_path: The full file path of the .DNG file to open

    Returns:
        The original capture date, as a datetime.
    '''

    dng_image_path_root, dng_image_extension = os.path.splitext(dng_image_path)
    raw_image_path = f'{dng_image_path_root}.jpeg'

    with open(raw_image_path, 'rb') as raw_image_file:
        tags = exifread.process_file(raw_image_file)

    date_taken = tags[EXIF_DATE_TIME_ORIGINAL_TAG]

    return datetime.datetime.strptime(date_taken.values, '%Y:%m:%d %H:%M:%S')
