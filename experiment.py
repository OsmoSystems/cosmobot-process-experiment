'''Process Image'''
import os
from shutil import copyfile, rmtree
from time import sleep
import argparse
from datetime import datetime, timedelta
from upload import upload_files
import camera

# construct the argument parser and parse the arguments
AP = argparse.ArgumentParser()
AP.add_argument("-d", "--duration", required=True, type=int, help="duration in seconds")
AP.add_argument("-i", "--interval", required=True, type=int, help="interval for image capture in seconds")
AP.add_argument("-n", "--name", required=True, type=str, help="name for experiment")
ARGS = vars(AP.parse_args())

DURATION = ARGS['duration']
INTERVAL = ARGS['interval']
EXP_NAME = ARGS['name']

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

def performExperiment():
    SEQUENCE = 1
    while datetime.now() < END_DATETIME:
        IMAGE_FILENAME = OUTPUT_FOLDER + "/{}.jpeg".format(SEQUENCE)
        camera.captureImage(IMAGE_FILENAME)

        SEQUENCE = SEQUENCE + 1
        sleep(INTERVAL)

    S3_BASE = ""
    S3_LOCATION = S3_BASE + OUTPUT_FOLDER

    print("Uploading Experiment to S3 - {}".format(S3_LOCATION))
    # upload to s3 the experiments that exist in output
    upload_files('./output')

    # remove only this current experiment
    # rmtree(OUTPUT_FOLDER)

performExperiment()
