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

from .file_structure import create_directory, iso_datetime_for_filename, get_base_output_path

DEFAULT_ISO = 100
DEFAULT_EXPOSURE = 1500000
DEFAULT_CAPTURE_PARAMS = f' -ss {DEFAULT_EXPOSURE} -ISO {DEFAULT_ISO}'

ExperimentConfiguration = namedtuple(
    'ExperimentConfiguration',
    [
        'name',  # The Name of the experiment.  Used for naming directories for output.
        'interval',  # The interval in seconds between the capture of images.
        'duration',  # How long in seconds should the experiment run for.
        'variants',  # array of ExperimentVariants that define different capture settings to be run each iteration
        'start_date',  # date the experiment was started
        'end_date',  # date at which to end the experiment.  If duration is not set then this is effectively indefinite
        'experiment_directory_path',  # directory/path to write files to
        'command',  # full command with arguments issued to start the experiment from the command line
        'git_hash',  # git hash of camera-sensor-prototype repo
        'hostname',  # hostname of the device the experient was executed on
        'mac',  # mac address
    ]
)

ExperimentVariant = namedtuple(
    'ExperimentVariant',
    [
        'capture_params',  # parameters to pass to raspistill binary through the command line
    ]
)


def _parse_args():
    '''Extract and verify arguments passed in from the command line
     Args:
        None
     Returns:
        dictionary of arguments parsed from the command line
    '''
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--name', required=True, type=str, help='name for experiment')
    arg_parser.add_argument('--interval', required=True, type=int, help='interval between image capture in seconds')
    arg_parser.add_argument('--duration', required=False, type=int, default=None, help='Duration in seconds. Optional.')
    arg_parser.add_argument('--variant', required=False, type=str, default=[], action='append',
                            help='variants of camera capture parameters to use during experiment.'
                            'Ex: --variant " -ss 500000 -ISO 100" --variant " -ss 100000 -ISO 200" ...'
                            f'If not provided, "{DEFAULT_CAPTURE_PARAMS}" will be used')

    arg_parser.add_argument("--exposures", required=False, type=int, nargs='+', default=None,
                            help="list of exposures to iterate capture through ex. --exposures 1000000, 2000000")
    arg_parser.add_argument("--isos", required=False, type=int, nargs='+', default=None,
                            help='List of isos to iterate capture through ex. --isos 100, 200 . \n'
                            f' If not provided and --exposures is provided, ISO {DEFAULT_ISO} will'
                            ' be used when iterating over exposures.')

    return vars(arg_parser.parse_args())


def get_experiment_variants(args):
    variants = [
        ExperimentVariant(capture_params=capture_params)
        for capture_params in args['variant']
    ]

    # add variants of exposure and iso lists if provided
    if args['exposures']:
        isos = args['isos'] or [DEFAULT_ISO]
        variants.extend(
            ExperimentVariant(capture_params=f'" -ss {exposure} -ISO {iso}"')
            for exposure in args['exposures']
            for iso in isos
        )

    if not variants:
        variants = [ExperimentVariant(capture_params=DEFAULT_CAPTURE_PARAMS)]

    return variants


def get_experiment_configuration():
    '''Return a constructed named experimental configuration in a namedtuple.
     Args:
        None, but retrieves arguments from the command line using _parse_args
     Returns:
        an instance of ExperimentConfiguration namedtuple

    '''
    args = _parse_args()

    duration = args['duration']
    start_date = datetime.now()
    end_date = start_date if duration is None else start_date + timedelta(seconds=duration)
    mac_address = hex(get_mac()).upper()
    mac_last_4 = str(mac_address)[-4:]

    iso_ish_datetime = iso_datetime_for_filename(start_date)
    experiment_directory_name = f'{iso_ish_datetime}-Pi{mac_last_4}-{args["name"]}'
    experiment_directory_path = os.path.join(get_base_output_path(), experiment_directory_name)

    variants = get_experiment_variants(args)

    experiment_configuration = ExperimentConfiguration(
        name=args['name'],
        interval=args['interval'],
        duration=duration,
        start_date=start_date,
        end_date=end_date,
        experiment_directory_path=experiment_directory_path,
        command=' '.join(sys.argv),
        git_hash=_git_hash(),
        hostname=gethostname(),
        mac=mac_address,
        variants=variants
    )

    return experiment_configuration


def create_file_structure_for_experiment(configuration):
    create_directory(configuration.experiment_directory_path)

    metadata_path = os.path.join(configuration.experiment_directory_path, 'experiment_metadata.yml')
    with open(metadata_path, 'w') as metadata_file:
        yaml.dump(configuration._asdict(), metadata_file, default_flow_style=False)


def hostname_is_valid(hostname):
    '''Does hostname follow the pattern we expect pi-cam-[last four of MAC]
     Args:
        hostname: hostname of machine
     Returns:
        Boolean: is hostname valid
    '''
    return re.search('^pi-cam-[0-9A-F]{4}$', hostname) is not None


def _git_hash():
    '''Retrieve git hash if it exists
     Args:
        None
     Returns:
        Git hash or error message
    '''
    command = 'git rev-parse HEAD'

    try:
        command_output = check_output(command, shell=True).decode('utf-8').rstrip()
    except CalledProcessError:
        command_output = '"git rev-parse HEAD" retrieval failed.  No repo?'

    return command_output
