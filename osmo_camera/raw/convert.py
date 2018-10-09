import os
from subprocess import call


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


def _folder_to_dng_recursive(raspiraw_location, raw_images_dir):
    # Convert .jpegs in folder recursively
    # NOTE: Currently recursive conversion is not our intended method of extracting
    # raw data to a DNG file.  (This was a one off request by Jacob for support)
    # TODO: Determine which to use - https://app.asana.com/0/823265982730077/858190003215284
    for root, _, files in os.walk(raw_images_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                full_path = os.path.join(root, file)
                _file_to_dng(raspiraw_location, full_path)


def to_dng(raspiraw_location, raw_image_path=None, raw_images_dir=None):
    if (not raw_image_path and not raw_images_dir) or (raw_image_path and raw_images_dir):
        raise Exception('Only one of "raw_image_path" or "raw_images_dir" can be defined.')

    if raw_images_dir:
        return _folder_to_dng(raspiraw_location, raw_images_dir)

    return _file_to_dng(raspiraw_location, raw_image_path)
