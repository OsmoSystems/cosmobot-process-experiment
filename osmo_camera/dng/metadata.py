import datetime
import os

import exifread


EXIF_DATE_TIME_ORIGINAL_TAG = 'EXIF DateTimeOriginal'


# TODO: stop assuming the jpeg version of the file is alongside the DNG file. Either copy over metadata to .DNG, or..?
def capture_date(dng_filename):
    ''' Extracts original capture date from a .DNG file

    Captue date is stored as an EXIF key formatted like:
        'EXIF DateTimeOriginal': (0x0132) ASCII=2018:09:10 20:01:19 @ 59140

    the right side is an EXIF key value; getting ex_key.values gives you a nice ISO8601-ish string

    Args:
        dng_filename: The name of the .DNG file to open

    Returns:
        The original capture date, as a datetime.


    '''

    filename_prefix, file_extension = os.path.splitext(dng_filename)
    raw_filename = f'{filename_prefix}.jpeg'

    with open(raw_filename, 'rb') as raw_file:
        tags = exifread.process_file(raw_file)

    date_taken = tags[EXIF_DATE_TIME_ORIGINAL_TAG]
    return datetime.datetime.strptime(date_taken.values, '%Y:%m:%d %H:%M:%S')
