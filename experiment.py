'''Perform camera capture experiment'''

import os
from time import sleep
import argparse
from datetime import datetime, timedelta
from camera import capture_image
from process_experiment import convert_img_in_dir_to_dng, s3_sync_output_dir

AP = argparse.ArgumentParser()
AP.add_argument("-d", "--duration", required=True, type=int, help="duration in seconds")
AP.add_argument("-i", "--interval",
                required=True,
                type=int,
                help="interval for image capture in seconds")
AP.add_argument("-n", "--name", required=True, type=str, help="name for experiment")
AP.add_argument("-c", "--convert_to_dng",
                required=False,
                nargs='?',
                type=bool,
                default=True,
                help="convert images to dng after experiment is complete, keep the original files")
AP.add_argument("-s", "--sync",
                required=False,
                nargs='?',
                type=bool,
                default=True,
                help="Sync to s3 after experiment is complete")
AP.add_argument("--addl_capture_params",
                required=False,
                type=str,
                help="additional parameters passed to raspistill when capturing images")

ARGS = vars(AP.parse_args())

DURATION = ARGS['duration']
INTERVAL = ARGS['interval']
EXP_NAME = ARGS['name']
ADDITIONAL_CAPTURE_PARAMS = ARGS['addl_capture_params']
SHOULD_CONVERT_TO_DNG = ARGS['convert_to_dng']
SHOULD_SYNC = ARGS['sync']

OUTPUT_FOLDER = datetime.now().strftime('./output/%Y%m%d%H%M%S_{}'.format(EXP_NAME))

START_DATETIME = datetime.now()
END_DATETIME = START_DATETIME + timedelta(seconds=DURATION)

if not os.path.exists(OUTPUT_FOLDER):
    print('creating folder: {}'.format(OUTPUT_FOLDER))
    os.makedirs(OUTPUT_FOLDER)


EXPERIMENT_DICT = dict(
    START_DATETIME=START_DATETIME,
    END_DATETIME=END_DATETIME,
    DURATION=DURATION,
    INTERVAL=INTERVAL,
    OUTPUT_FOLDER=OUTPUT_FOLDER
)


def perform_experiment():
    '''
    Perform while loop while time is less than END_DATETIME
    (seconds from start of program passed in duration arg)
    '''

    # image sequence during camera capture
    sequence = 1

    while datetime.now() < END_DATETIME:
        image_filename = OUTPUT_FOLDER + "/{}.jpeg".format(sequence)
        capture_image(image_filename, additional_capture_params=ADDITIONAL_CAPTURE_PARAMS)

        sequence = sequence + 1
        sleep(INTERVAL)

    if SHOULD_CONVERT_TO_DNG is not None:
        convert_img_in_dir_to_dng(OUTPUT_FOLDER)

    if SHOULD_SYNC is not None:
        s3_sync_output_dir()


perform_experiment()
