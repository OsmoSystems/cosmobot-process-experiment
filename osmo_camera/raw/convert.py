import os


def to_dng(raw_image_filename):
    # TODO: actually do the conversion instead of assuming converted file already exists
    # TODO: correctly manage camera v1 vs v2?
    filename_prefix, file_extension = os.path.splitext(raw_image_filename)
    return f'{filename_prefix}.dng'


# TODO: remove this once it's being done programmatically
def generate_raw_to_dng_conversion_command(source_files_folder, raspi_raw_location):
    raspi_raw_command = os.path.join(raspi_raw_location, 'raspiraw/raspi_dng_sony')

    for file in os.listdir(source_files_folder):
        input_file = os.path.join(source_files_folder, file)

        file_basename, dot, extension = file.partition('.')
        if extension != 'jpeg':
            # This is not one of our files
            continue
        output_filename = f'{file_basename}.dng'
        output_file = os.path.join(source_files_folder, output_filename)

        print(raspi_raw_command, f'"{input_file}"', f'"{output_file}"')


if __name__ == '__main__':
    raspi_raw_location = '/Users/jaime/osmo'
    source_files_folder = '/Users/jaime/osmo/cosmobot-data-sets/20180918183008_controled_pi_DO_test_7_picam2lowexposure'
    generate_raw_to_dng_conversion_command(source_files_folder, raspi_raw_location)
