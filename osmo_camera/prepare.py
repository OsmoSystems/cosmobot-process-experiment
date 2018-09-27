'''Perform camera capture experiment'''
import argparse
import os
import re
import math
from socket import gethostname
from shutil import disk_usage
from datetime import datetime, timedelta
from subprocess import check_output, CalledProcessError

# used if no duration is provided and for comparison in experiment runner
ONE_YEAR_IN_SECONDS = 31536000


def experiment_configuration(args):
    '''Extract and verify arguments passed in from the command line and build a
    dictionary of values that define an experiments configuration.
     Args:
        args: arguments passed in through the command line
     Returns:
        an experimental configuration dictionary with the following keys
          interval (required): The interval in seconds between the capture of images.
          name (required): The Name of the experiment.  Used for naming folders for output.
          duration (optional): How long in seconds should the experiment run for.
          variants (optional): array of variants that define different capture settings to
          be run during each capture iteration
          command (retrieved): full command issued from the command line
          git_hash (retrieved): git hash or message that says no git repo present
    '''

    base_output_path = '../output/'

    # initailize configuration dictionary with command issued and git_hash (if available)
    configuration = dict(
        command=' '.join(args),
        git_hash=git_hash(),
        hostname=gethostname(),
    )

    arg_parser = argparse.ArgumentParser()

    # required arguments
    arg_parser.add_argument("--interval",
                            required=True,
                            type=int,
                            help="interval for image capture in seconds")
    arg_parser.add_argument("--name",
                            required=True,
                            type=str,
                            help="name for experiment")
    # optional arguments
    arg_parser.add_argument("--duration",
                            required=False,
                            type=int,
                            default=ONE_YEAR_IN_SECONDS,
                            help="duration in seconds")
    arg_parser.add_argument("--capture_params",
                            required=False,
                            type=str,
                            help="additional parameters passed to raspistill when capturing " +
                            "images. example: --capture_params ' -ss 500000 -iso 100'")
    arg_parser.add_argument("--variant",
                            required=False,
                            type=str,
                            action='append',
                            nargs=2,
                            metavar=('name', 'capture_params'),
                            help="variants of image capture to use during experiment." +
                            "example: --variant capture_type1 ' -ss 500000 -iso 100' " +
                            "--variant capture_type2 ' -ss 100000 -iso 200' ...")

    args = vars(arg_parser.parse_args())

    # required args

    # start_date of experiment is now
    start_date = datetime.now()


    # There is always at least one variant for an experiment if name and interval are provided
    variants = []
    variants.append(dict(
        name=args['name'],
        capture_params=args['capture_params'],
        output_folder=base_output_path + start_date.strftime(
            '%Y%m%d%H%M%S_{}'.format(args['name']))
    ))

    # extract the variants passed through the command line (if any)
    variants_from_cmd = args['variant']

    # if additional variants are specified add them to the list
    if variants_from_cmd:
        for _, variant in enumerate(variants_from_cmd):
            # TODO: less magic way to do extract arg tuples?
            variant_name = variant[0]
            variant_dict = dict(
                name=variant_name,
                capture_params=variant[1],
                output_folder=base_output_path + start_date.strftime(
                    '%Y%m%d%H%M%S_{}'.format(variant_name))
            )
            variants.append(variant_dict)

    configuration['variants'] = variants
    configuration['interval'] = args['interval']
    configuration['name'] = args['name']
    configuration['start_date'] = start_date
    configuration['duration'] = args['duration']
    configuration['end_date'] = start_date + timedelta(seconds=args['duration'])

    return configuration


def create_output_folder(folder_name, base_path='../output/'):
    '''Create a folder if it does not exist'''
    folder_to_create = base_path + folder_name
    if not os.path.exists(folder_to_create):
        print('creating folder: {}'.format(folder_to_create))
        os.makedirs(folder_to_create)
    else:
        print('folder {} already exists'.format(folder_to_create))


def is_hostname_valid(hostname):
    '''Does hostname follow the pattern we expect pi-cam-[last four of MAC]
     Args:
        hostname: hostname of machine
     Returns:
        Boolean: is hostname valid
    '''
    if not re.search("[0-9]{4}$", hostname) and not re.search("pi_cam", hostname):
        return False

    return True


# Experimental evidence shows the raw image size on the Sony IMX Camera module
# to max out at 1600000 bytes in size
IMAGE_SIZE_IN_BYTES = 1600000

CAPTURE_TIME = 5


def estimate_image_count(duration, interval):
    '''Estimate how many images will be captured with interval and duration
     Args:
        duration: seconds the experiment will run for
        interval: interval in seconds between image capture
     Returns:
        float: How many images can be stored
    '''
    return int(math.floor(duration / CAPTURE_TIME))


def how_many_images_with_free_space():
    '''Estimate how many images can be stored on the storage device
     Args:
        None
     Returns:
        Boolean: How many images can be stored
    '''
    _, _, free = disk_usage('/')
    return free / IMAGE_SIZE_IN_BYTES


def free_space_for_image_count(image_count):
    '''Check if there is enough space with the storage device
     Args:
        image_count: how many images will be stored
     Returns:
        Boolean: True/False - is there space to store the experiment
    '''
    _, _, free = disk_usage('/')
    return free > IMAGE_SIZE_IN_BYTES * image_count


def free_space_for_one_image():
    '''Is there enough space for one image
     Args:
        None
     Returns:
        Boolean: True/False - is there space to store one image
    '''
    return free_space_for_image_count(1)


def free_space_for_experiment(duration, interval):
    '''Is there enough space for the entire experiment
     Args:
        duration: seconds the experiment will run for
        interval: interval in seconds between image capture
     Returns:
        Boolean: True/False - is there space to store experiment
    '''
    # TODO: assumes capture takes no time
    image_count = duration / interval * IMAGE_SIZE_IN_BYTES
    return free_space_for_image_count(image_count)


def git_hash():
    '''Retrieve hit hash if it exists
     Args:
        None
     Returns:
        Boolean: git hash or if not git repo error message
    '''
    comm = 'git rev-parse HEAD'

    try:
        comm_output = check_output(comm, shell=True).decode("utf-8").rstrip()
    except CalledProcessError:
        comm_output = "'git rev-parse HEAD' retrieval failed.  No repo?"

    return comm_output
