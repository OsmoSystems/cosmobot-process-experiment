import argparse
import sys
import os
import re
from socket import gethostname
from datetime import datetime
from subprocess import check_output, CalledProcessError


def experiment_configuration(base_output_path='../output/'):
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
          be run during each capture iteration.  a variant requires two sub arguments
            name (variant[0]): name to identify variant with e.g. 'short_exposure' or 'long_exposure'
            capture_params (variant[1]): parameters to pass to raspistill command e.g.
                                         ' -ss 100 -iso 100'
          command (retrieved): full command issued from the command line
          git_hash (retrieved): git hash or message that says no git repo present
    '''

    # initailize configuration dictionary with command issued and git_hash (if available)
    configuration = dict(
        command=' '.join(sys.argv),
        git_hash=_git_hash(),
        hostname=gethostname(),
    )

    arg_parser = argparse.ArgumentParser()

    # required arguments
    arg_parser.add_argument("--interval", required=True, type=int, help="interval for image capture in seconds")
    arg_parser.add_argument("--name", required=True, type=str, help="name for experiment")
    arg_parser.add_argument("--variant", required=True, type=str, action='append', nargs=2,
                            metavar=('name', 'capture_params'),
                            help="variants of image capture to use during experiment." +
                            "example: --variant capture_type1 ' -ss 500000 -iso 100' " +
                            "--variant capture_type2 ' -ss 100000 -iso 200' ...")

    # optional arguments
    arg_parser.add_argument("--duration", required=False, type=int, default=None,
                            help="duration in seconds")

    args = vars(arg_parser.parse_args())

    # start_date of experiment is now
    start_date = datetime.now()

    variants = []

    configuration['interval'] = args['interval']
    configuration['name'] = args['name']
    configuration['start_date'] = start_date
    configuration['duration'] = args['duration']

    experiment_output_folder = base_output_path + start_date.strftime(f'%Y%m%d%H%M%S_{args["name"]}')
    configuration['experiment_output_folder'] = experiment_output_folder
    _create_output_folder(experiment_output_folder)

    # add variants to the list of variants
    for _, variant in enumerate(args['variant']):
        variant_name = variant[0]
        variant_dict = dict(
            name=variant_name,
            capture_params=variant[1],
            output_folder=experiment_output_folder + f'/{variant_name}',
            metadata=configuration
        )
        _create_output_folder(variant_dict["output_folder"])
        variants.append(variant_dict)

    configuration['variants'] = variants

    return configuration


def _create_output_folder(folder_name, base_path='../output/'):
    '''Create a folder if it does not exist'''
    folder_to_create = base_path + folder_name
    if not os.path.exists(folder_to_create):
        print(f'creating folder: {folder_to_create}')
        os.makedirs(folder_to_create)
    else:
        print(f'folder {folder_to_create} already exists')


def is_hostname_valid(hostname):
    '''Does hostname follow the pattern we expect pi-cam-[last four of MAC]
     Args:
        hostname: hostname of machine
     Returns:
        Boolean: is hostname valid
    '''
    if re.search("[0-9]{4}$", hostname) and re.search("pi-cam", hostname):
        return True

    return False


def _git_hash():
    '''Retrieve hit hash if it exists
     Args:
        None
     Returns:
        Boolean: git hash or if not git repo error message
    '''
    command = 'git rev-parse HEAD'

    try:
        command_output = check_output(command, shell=True).decode("utf-8").rstrip()
    except CalledProcessError:
        command_output = "'git rev-parse HEAD' retrieval failed.  No repo?"

    return command_output
