import os
from datetime import datetime, timedelta
from camera import capture
from file_structure import iso_datetime_for_filename
from prepare import hostname_is_valid, get_experiment_configuration, create_file_structure_for_experiment
from storage import how_many_images_with_free_space, free_space_for_one_image
from sync_manager import sync_directory_in_separate_process, end_syncing_processes


def perform_experiment(configuration):
    '''Perform experiment using settings passed in through the configuration.
       experimental configuration defines the capture frequency and duration of the experiment
       as well as controlling the camera settings to be used to capture images.
       Experimental output directories are created prior to initiating image capture and
       experimental metadata is collected during the experiment and saved.
       Finally, imagery and experimental metadata is synced to s3 on an ongoing basis.
     Args:
        configuration: ExperimentConfiguration instance. Determines how the experiment should be performed.
     Returns:
        None

     Notes on local development:
       There is a helper function to simulate a capture of a file by copying it into
       the location a capture would place a file.  You can use it by changing the from
       from camera import capture => from camera import capture, simulate_capture_with_copy
       and using simulate_capture_with_copy instead of capture.
    '''

    # print out warning that no duration has been set and inform how many
    # estimated images can be stored
    if configuration.duration is None:
        how_many_images_can_be_captured = how_many_images_with_free_space()
        print("No experimental duration provided.")
        print(f"Estimated number of images that can be captured with free space: {how_many_images_can_be_captured}")

    # Initial value of start_date results in immediate capture on first iteration in while loop
    next_capture_time = configuration.start_date

    # image sequence during camera capture
    sequence = 1

    while configuration.duration is None or datetime.now() < configuration.end_date:
        if datetime.now() < next_capture_time:
            continue

        # next_capture_time is agnostic to the time needed for capture and writing of image
        next_capture_time = next_capture_time + timedelta(seconds=configuration.interval)

        # iterate through each capture variant and capture an image with it's settings
        for variant in configuration.variants:

            if not free_space_for_one_image():
                quit("There is insufficient space to save the image.  Quitting.")

            iso_ish_datetime = iso_datetime_for_filename(configuration.start_date)
            image_filename = f'{iso_ish_datetime}-{sequence}.jpeg'
            image_filepath = os.path.join(variant.output_directory, image_filename)

            capture(image_filepath, additional_capture_params=variant.capture_params)

            # this may do nothing depending on if sync is currently occuring
            sync_directory_in_separate_process(variant.output_directory)

        sequence = sequence + 1

    final_sync_for_experiment(configuration.variants)


def final_sync_for_experiment(variants):
    # final sync when experiment runner finishes or a keyboard interrupt is detected
    # From testing I noticed that if a file(s) is written during after a sync process begins it was
    # not being added to a list to sync. My hunch is that this is due to when a syncing process initially begins,
    # it compares a local list with the remote list and will keep those lists in memory. If additional files are
    # written after a syncing process begins they will not sync.  so, to finish things up we shut down all of the
    # existing sync processes and start new ones
    end_syncing_processes()
    for variant in variants:
        sync_directory_in_separate_process(variant.output_directory, wait_for_finish=True)


if __name__ == '__main__':
    configuration = get_experiment_configuration()
    create_file_structure_for_experiment(configuration)

    if hostname_is_valid(configuration.hostname):
        QUIT_MESSAGE = f'"{configuration.hostname}" is not a valid hostname.'
        QUIT_MESSAGE += ' Contact your local dev for instructions on setting a valid hostname.'

    try:
        perform_experiment(configuration)
    except KeyboardInterrupt:
        print('Keyboard interrupt detected, attempting final sync')
        final_sync_for_experiment(configuration.variants)
        quit('Final sync after keyboard interrupt completed.')
