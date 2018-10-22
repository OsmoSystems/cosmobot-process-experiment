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

from .file_structure import create_directory, iso_datetime_for_filename

BASE_OUTPUT_PATH = os.path.abspath('../output/')

ExperimentConfiguration = namedtuple(
    'ExperimentConfiguration',
    [
        'name',  # The Name of the experiment.  Used for naming directories for output.
        'interval',  # The interval in seconds between the capture of images.
        'duration',  # How long in seconds should the experiment run for.
        'variants',  # array of variants that define different capture settings to be run during each capture iteration
        'start_date',  # date the experiment was started
        'end_date',  # date at which to end the experiment.  If duration is not set then this is effectively indefinite
        'experiment_directory_path',  # directory/path to write files to
        'command',  # full command with arguments issued to start the experiment from the command line
        'git_hash',  # git hash of camera-sensor-prototype repo
        'hostname',  # hostname of the device the experient was executed on
        'mac',  # mac address
        'mac_last_4'  # last four of mac address
    ]
)

ExperimentVariant = namedtuple(
    'ExperimentVariant',
    [
        'name',  # name of variant
        'capture_params',  # parameters to pass to raspistill binary through the command line
        'output_directory'  # (deprecated) output directory for the variant within the top level experiment directory
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

    # required arguments
    arg_parser.add_argument("--interval", required=True, type=int, help="interval for image capture in seconds")
    arg_parser.add_argument("--name", required=True, type=str, help="name for experiment")
    arg_parser.add_argument("--variant", required=True, type=str, action='append', nargs=1,
                            metavar=('name', 'capture_params'),
                            help="variants of image capture to use during experiment." +
                            "example: --variant ' -ss 500000 -iso 100' " +
                            "--variant ' -ss 100000 -iso 200' ...")

    # optional arguments
    arg_parser.add_argument("--duration", required=False, type=int, default=None,
                            help="duration in seconds")

    settings_group = arg_parser.add_argument_group('settings')
    settings_group.add_argument("--exposures", required=False, type=int, nargs='+', default=None,
                                help="list of exposures to iterate capture through ex. --exposures 1000000, 2000000")
    settings_group.add_argument("--isos", required=False, type=int, nargs='+', default=None,
                                help="list of isos to iterate capture through ex. --isos 100, 200")

    return vars(arg_parser.parse_args())


def _parse_variants_from_settings_lists(exposures, isos):
    return [
        f'" -iso {iso} -ss {exposure}" '
        for exposure in exposures
        for iso in isos
    ]


def get_experiment_configuration():
    '''Return a constructed named experimental configuration in a namedtuple.
     Args:
        None, but retrieves arguments from the command line using _parse_args
     Returns:
        an instance of ExperimentConfiguration namedtuple

    '''
    args = _parse_args()
    start_date = datetime.now()
    end_date = start_date if args['duration'] is None else start_date + timedelta(seconds=args['duration'])

    iso_ish_datetime = iso_datetime_for_filename(start_date)

    experiment_directory_name = f'{iso_ish_datetime}-MAC{str(get_mac())[-4:]}-{args["name"]}'
    experiment_directory_path = os.path.join(BASE_OUTPUT_PATH, experiment_directory_name)

    experiment_configuration = ExperimentConfiguration(
        name=args['name'],
        interval=args['interval'],
        duration=args['duration'],
        variants=[],
        start_date=start_date,
        end_date=end_date,
        experiment_directory_path=experiment_directory_path,
        command=' '.join(sys.argv),
        git_hash=_git_hash(),
        hostname=gethostname(),
        mac=get_mac(),
        mac_last_4=str(get_mac())[-4:]
    )

    # add variants to the list of variants
    for variant in args['variant']:
        capture_params = variant
        variant_name = capture_params.replace(' ', '_').replace('-', '')
        experiment_configuration.variants.append(
            ExperimentVariant(
                name=variant_name,
                capture_params=capture_params,
                output_directory=experiment_directory_path
            )
        )

    if args['exposures'] is not None and args['isos'] is not None:
        experiment_configuration.variants.append(
            [
                ExperimentVariant(
                    name=f'" -ISO {iso} -ss {exposure}" ',
                    capture_params='',
                    output_directory=experiment_directory_path
                )
                for exposure in args['exposures']
                for iso in args['isos']
            ]
        )

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


def hostname_is_valid(hostname):
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
