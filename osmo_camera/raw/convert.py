import os
from subprocess import call


# TODO: don't depend on knowing where raspiraw lives?
# TODO: correctly manage camera v1 vs v2?
def file_to_dng(raw_image_path, raspi_raw_location):
    raw_image_root, raw_image_extension = os.path.splitext(raw_image_path)
    output_dng_image_path = f'{raw_image_root}.dng'

    raspi_raw_command = os.path.join(raspi_raw_location, 'raspiraw/raspi_dng_sony')

    command = f'{raspi_raw_command} "{raw_image_path}" "{output_dng_image_path}"'
    call([command], shell=True)

    return output_dng_image_path


def folder_to_dng(raw_images_dir, raspi_raw_location):
    for raw_image_filename in os.listdir(raw_images_dir):
        raw_image_path = os.path.join(raw_images_dir, raw_image_filename)

        file_to_dng(raw_image_path, raspi_raw_location)
