'''Perform camera capture experiment'''

import os
import sys

import argparse
from datetime import datetime, timedelta
from camera import capture_image
from process_experiment import s3_sync_output_dir_synchronous, s3_sync_output_dir_asynchronously

def parse_args_for_experiment_configuration(args):
    experiment_configuration = dict()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--duration",
                            required=False,
                            type=int,
                            help="duration in seconds")
    arg_parser.add_argument("-i", "--interval",
                            required=True,
                            type=int,
                            help="interval for image capture in seconds")
    arg_parser.add_argument("-n",
                            "--name",
                            required=True,
                            type=str,
                            help="name for experiment")
    arg_parser.add_argument("-c", "--convert_to_dng",
                            required=False,
                            nargs='?',
                            type=bool,
                            default=True,
                            help="Convert images to dng once experiment is complete")
    arg_parser.add_argument("-s", "--sync",
                            required=False,
                            nargs='?',
                            type=bool,
                            default=True,
                            help="Sync to s3 after experiment is complete")
    arg_parser.add_argument("-cs", "--continuous-sync",
                            required=False,
                            nargs='?',
                            type=bool,
                            default=True,
                            help="Sync to s3 after experiment is complete")
    arg_parser.add_argument("--addl_capture_params",
                            required=False,
                            type=str,
                            help="additional parameters passed to raspistill when capturing images")
    arg_parser.add_argument("--retain-images",
                            required=False,
                            type=int,
                            help="Retain images that are produced through an experiment")
    arg_parser.add_argument("--variant",
                            required=False,
                            type=str,
                            action='append',
                            nargs=2,
                            metavar=('variant-name', 'addl_capture_params'),
                            help="variant capture methods to run during experiment")

    args = vars(arg_parser.parse_args())


    # required args
    name = args['name']
    experiment_configuration['interval'] = args['interval']
    experiment_configuration['name'] = name

    # optional args
    experiment_configuration['addl_capture_params'] = args['addl_capture_params']
    experiment_configuration['convert_to_dng'] = args['convert_to_dng']
    experiment_configuration['sync'] = args['sync']
    experiment_configuration['continuous_sync'] = args['continuous_sync']
    experiment_configuration['variant'] = args['variant']

    # if no duration set then run for an arbitrary long period (a year)
    seconds_in_year = 31536000
    duration = args['duration'] if args['duration'] is not None else seconds_in_year

    start_date = datetime.now()
    output_folder = start_date.strftime('./output/%Y%m%d%H%M%S_{}'.format(name))

    experiment_configuration['output_folder'] = output_folder
    experiment_configuration['start_date'] = start_date
    experiment_configuration['end_date'] = start_date + timedelta(seconds=duration)

    return experiment_configuration

def perform_experiment(experiment_configuration):
    '''
    Perform while loop while time is less than END_DATETIME
    (seconds from start of program passed in duration arg)
    '''

    output_folder, start_date, end_date, interval, addl_capture_params = experiment_configuration
    continuous_sync, sync = experiment_configuration

    if not os.path.exists(output_folder):
        print('creating folder: {}'.format(output_folder))
        os.makedirs(output_folder)

    # Date at which next image capture should occur.
    # Initial value of start_date results in immediate capture on first iteration in while loop
    next_capture_time = start_date

    # image sequence during camera capture
    sequence = 1

    while datetime.now() < end_date and datetime.now() > next_capture_time:
        image_filename = start_date.strftime('/%Y%m%d%H%M%S_{}.jpeg'.format(sequence))
        image_filename = output_folder + image_filename

        capture_image(image_filename, additional_capture_params=addl_capture_params)

        sequence = sequence + 1
        next_capture_time = next_capture_time + timedelta(seconds=interval)

        if continuous_sync:
            s3_sync_output_dir_asynchronously()

    if sync is not None:
        s3_sync_output_dir_synchronous()

if __name__ == '__main__':
    exp_config = parse_args_for_experiment_configuration(sys.argv)
    print(exp_config)
    perform_experiment(exp_config)
