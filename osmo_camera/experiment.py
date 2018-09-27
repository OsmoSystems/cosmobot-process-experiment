'''Perform camera capture experiment'''

import sys
import yaml
from datetime import datetime, timedelta
from camera import capture
from prepare import create_output_folder, is_hostname_valid, experiment_configuration
from prepare import how_many_images_with_free_space, free_space_for_one_image, ONE_YEAR_IN_SECONDS
from sync import sync_directory_in_separate_process


def perform_experiment(configuration):
    '''Perform experiment using settings passed in through the configuration.
       experimental configuration defines the capture frequency and duration of the experiment
       as well as controlling the camera settings to be used to capture images.
       Experimental output folders are created prior to initiating image capture and
       experimental metadata is collected during the experiment and saved.
       Finally, imagery and experimental metadata is synced to s3 on an ongoing basis.
     Args:
        configuration: dictionary containing values that define how an experiment
        should be performed.
     Returns:
        None
    '''

    # unpack experiment configuration variables
    interval = configuration["interval"]
    start_date = configuration["start_date"]
    end_date = configuration["end_date"]
    variants = configuration["variants"]
    duration = configuration["duration"]

    # print out information message that
    if duration == ONE_YEAR_IN_SECONDS:
        how_many_images_can_be_captured = how_many_images_with_free_space()
        print("No experimental duration provided.")
        print("Estimated number of images that can be captured with free space: {}"
              .format(how_many_images_can_be_captured))

    # create output folders for each variant
    for _, variant in enumerate(configuration["variants"]):
        create_output_folder(variant["output_folder"])
        # set metadata to the experiment configuration dictionary in order
        # for each variant to have output of the entire configuration
        variant["metadata"] = configuration

    # Initial value of start_date results in immediate capture on first iteration in while loop
    next_capture_time = start_date

    # image sequence during camera capture
    sequence = 1

    while datetime.now() < end_date:
        if datetime.now() > next_capture_time:

            # iterate through each capture variant and capture an image with it's settings
            for _, variant in enumerate(variants):

                # TODO: needed?
                if not free_space_for_one_image():
                    quit("There is insufficeint space to save the image.  Quitting.")

                # unpack variant values
                output_folder = variant['output_folder']
                capture_params = variant['capture_params']
                image_filename = start_date.strftime('/%Y%m%d%H%M%S_{}.jpeg'.format(sequence))
                image_filepath = output_folder + image_filename
                metadata_path = output_folder + '/experiment_metadata.yml'

                begin_date_for_capture = datetime.now()
                capture_info = capture(image_filepath, additional_capture_params=capture_params)
                ms_for_capture = (datetime.now() - begin_date_for_capture).microseconds

                metadata = dict(
                    ms_for_capture=ms_for_capture,
                    capture_info=capture_info
                )

                # for each image store a separate set of metadata with time for capture
                # and the capture info provided by raspistill
                variant["metadata"][image_filename] = metadata

                # write latest metadata for variant to yaml file
                with open(metadata_path, 'w') as outfile:
                    yaml.dump(variant["metadata"], outfile, default_flow_style=False)

                # this may do nothing depending on if sync is currently occuring
                sync_directory_in_separate_process(output_folder)

            sequence = sequence + 1
            next_capture_time = next_capture_time + timedelta(seconds=interval)

    # finally, for each variant/folder issue a final sync command
    for _, variant in enumerate(variants):
        sync_directory_in_separate_process(variant["output_folder"], final_sync=True)


if __name__ == '__main__':
    CONFIGURATION = experiment_configuration(sys.argv)
    HOSTNAME = CONFIGURATION['hostname']

    if is_hostname_valid(HOSTNAME):
        QUIT_MESSAGE = "\"" + HOSTNAME + "\" is not a valid hostname."
        QUIT_MESSAGE += " Contact your local dev for instructions on setting a valid hostname."
        quit(QUIT_MESSAGE)
    perform_experiment(CONFIGURATION)
