import sys
from datetime import datetime, timedelta
import yaml
from camera import capture, emulate_capture_with_copy
from prepare import is_hostname_valid, experiment_configuration
from storage import how_many_images_with_free_space, free_space_for_one_image
from sync_manager import sync_directory_in_separate_process, end_syncing_processes


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
    variants = configuration["variants"]
    duration = configuration["duration"]
    end_date = start_date if duration is None else start_date + timedelta(seconds=duration)

    # print out warning that no duration has been set and inform how many
    # estimated images can be stored
    if duration is None:
        how_many_images_can_be_captured = how_many_images_with_free_space()
        print("No experimental duration provided.")
        print(f"Estimated number of images that can be captured with free space: {how_many_images_can_be_captured}")

    # Initial value of start_date results in immediate capture on first iteration in while loop
    next_capture_time = start_date

    # image sequence during camera capture
    sequence = 1

    while duration is None or datetime.now() < end_date:
        if datetime.now() > next_capture_time:
            # next_capture_time is agnostic to the time needed for capture and writing of image
            next_capture_time = next_capture_time + timedelta(seconds=interval)

            # iterate through each capture variant and capture an image with it's settings
            for _, variant in enumerate(variants):

                if not free_space_for_one_image():
                    quit("There is insufficient space to save the image.  Quitting.")

                # unpack variant values
                output_folder = variant['output_folder']
                capture_params = variant['capture_params']
                image_filename = start_date.strftime(f'/%Y%m%d%H%M%S_{sequence}.jpeg')
                image_filepath = output_folder + image_filename
                metadata_path = output_folder + '/experiment_metadata.yml'

                begin_date_for_capture = datetime.now()

                # TODO: remove emulation (here for demonstration purposes)
                # capture_info = capture(image_filepath, additional_capture_params=capture_params)
                capture_info = emulate_capture_with_copy(image_filepath)

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

    end_syncing_processes()

    # finally, for each variant/folder issue a final sync command
    for _, variant in enumerate(variants):
        sync_directory_in_separate_process(variant["output_folder"], final_sync=True)


if __name__ == '__main__':
    CONFIGURATION = experiment_configuration(in_args=sys.argv)
    HOSTNAME = CONFIGURATION['hostname']

    if is_hostname_valid(HOSTNAME):
        QUIT_MESSAGE = "\"" + HOSTNAME + "\" is not a valid hostname."
        QUIT_MESSAGE += " Contact your local dev for instructions on setting a valid hostname."
        quit(QUIT_MESSAGE)

    perform_experiment(CONFIGURATION)
