import argparse
import sys
import os
import re
import yaml
from socket import gethostname
from datetime import datetime, timedelta
from subprocess import check_output, CalledProcessError
from uuid import getnode as get_mac
from collections import namedtuple
from osmo_camera.file_structure import create_directory, iso_datetime_for_filename

BASE_OUTPUT_PATH = os.path.abspath('../output/')
ExperimentConfiguration = namedtuple('ExperimentConfiguration', 'name interval duration variants start_date end_date experiment_directory_path command git_hash hostname mac mac_last_4')  # noqa: C0301, E501
ExperimentVariant = namedtuple('ExperimentVariant', 'name capture_params output_directory')


def _parse_args():
    '''Extract and verify arguments passed in from the command line
     Args:
        None
     Returns:
        dictionary of arguments parsed from the command line
    '''
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

    return vars(arg_parser.parse_args())


def get_experiment_configuration():
    '''Return a constructed named experimental configuration in a namedtuple.
     Args:
        None, but retrieves arguments from the command line using _parse_args
     Returns:
        an experiment configuration namedtuple
          interval (required): The interval in seconds between the capture of images.
          name (required): The Name of the experiment.  Used for naming directories for output.
          duration (optional): How long in seconds should the experiment run for.
          variants (optional): array of variants that define different capture settings to
          be run during each capture iteration.  a variant requires two sub arguments
            name (variant[0]): name to identify variant with e.g. 'short_exposure' or 'long_exposure'
            capture_params (variant[1]): parameters to pass to raspistill command e.g.
                                         ' -ss 100 -iso 100'
          command (retrieved): full command issued from the command line
          git_hash (retrieved): git hash or message that says no git repo present
    '''
    args = _parse_args()
    mac_address = get_mac()
    mac_last_4 = str(mac_address)[-4:]
    interval = args['interval']
    name = args['name']
    start_date = datetime.now()
    duration = args['duration']
    end_date = start_date if duration is None else start_date + timedelta(seconds=duration)

    iso_ish_datetime = iso_datetime_for_filename(start_date)

    experiment_directory_name = f'{iso_ish_datetime}-MAC{mac_last_4}-{args["name"]}'
    experiment_directory_path = os.path.join(BASE_OUTPUT_PATH, experiment_directory_name)

    experiment_configuration = ExperimentConfiguration(
        name,
        interval,
        duration,
        [],
        start_date,
        end_date,
        experiment_directory_path,
        ' '.join(sys.argv),
        _git_hash(),
        gethostname(),
        mac_address,
        mac_last_4
    )

    # add variants to the list of variants
    for variant in args['variant']:
        variant_name = variant[0]
        capture_params = variant[1]
        output_directory = os.path.join(experiment_configuration.experiment_directory_path, variant_name)
        experiment_configuration.variants.append(ExperimentVariant(variant_name, capture_params, output_directory))

    return experiment_configuration


def create_file_structure_for_experiment(configuration):
    create_directory(configuration.experiment_directory_path)

    # create variant directories and write experiment configuration metadata to file
    for variant in configuration.variants:
        create_directory(variant.output_directory)
        variant_output_directory = variant.output_directory
        metadata_path = os.path.join(variant_output_directory, 'experiment_metadata.yml')
        with open(metadata_path, 'w') as outfile:
            yaml.dump(configuration._asdict(), outfile, default_flow_style=False)


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
