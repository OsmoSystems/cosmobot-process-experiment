from osmo_camera import prepare
import argparse
from pytest import raises
import sys
# from prepare import experiment_configuration, create_output_folder,
# from prepare import estimate_image_count, images_with_free_space, free_space_for_image_count
# from prepare import free_space_for_one_image, free_space_for_experiment, git_hash

VALID_ARGS = ['experiment.py', '--interval', '10', '--name', 'exp1', '--variant', 'variant1', ' -ss 100 -iso 100']
INVALID_ARGS = ['experiment.py', '--interval', '10']


def test_command_args_are_valid():
    configuration = dict()

    configuration = prepare.experiment_configuration(VALID_ARGS)

    assert configuration['interval'] == 10
    assert configuration['name'] == 'exp1'
    assert configuration['variants'] == [{
        'name': 'variant1',
        'capture_params': ' -ss 100 -iso 100',
        'output_folder': '../output/20181004061249_exp1/variant1',
        'metadata': {...}
    }]


def test_command_args_are_invalid():
    with raises(SystemExit) as exception:
        prepare.experiment_configuration(INVALID_ARGS)

    assert exception is not None


def test_is_hostname_valid():
    assert prepare.is_hostname_valid('pi-cam-2222') is True


def test_is_hostname_invalid():
    assert prepare.is_hostname_valid('I am a naughty hostname') is False
