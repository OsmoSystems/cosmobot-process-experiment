import os
from subprocess import call


# TODO: don't depend on knowing where raspiraw lives?
# TODO: correctly manage camera v1 vs v2?
def _file_to_dng(raspiraw_location, raw_image_path):
    raw_image_root, raw_image_extension = os.path.splitext(raw_image_path)

    output_dng_image_path = f'{raw_image_root}.dng'

    # Don't convert if .DNG already exists
    if os.path.isfile(output_dng_image_path):
        return output_dng_image_path

    raspi_raw_command = os.path.join(raspiraw_location, 'raspiraw/raspi_dng_sony')
    command = f'{raspi_raw_command} "{raw_image_path}" "{output_dng_image_path}"'
    call([command], shell=True)

    return output_dng_image_path


def _folder_to_dng(raspiraw_location, raw_images_dir):
    # Only convert .jpegs (which are JPEG+RAW files)
    raw_image_paths = [
        os.path.join(raw_images_dir, filename)
        for filename in os.listdir(raw_images_dir)
        if filename.endswith('.jpeg')
    ]

    for raw_image_path in raw_image_paths:
        _file_to_dng(raspiraw_location, raw_image_path)


def to_dng(raspiraw_location, raw_image_path=None, raw_images_dir=None):
    if (not raw_image_path and not raw_images_dir) or (raw_image_path and raw_images_dir):
        raise Exception('Only one of "raw_image_path" or "raw_images_dir" can be defined.')

    if raw_images_dir:
        return _folder_to_dng(raspiraw_location, raw_images_dir)

    return _file_to_dng(raspiraw_location, raw_image_path)
