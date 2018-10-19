from subprocess import check_output
import argparse


def _parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exposures", required=True, type=int, nargs='+', default=None,
                            help="list of exposures to iterate capture through ex. --exposures [1000000, 2000000]")
    arg_parser.add_argument("--isos", required=True, type=int, nargs='+', default=None,
                            help="list of isos to iterate capture through ex. --isos [100, 200]")
    arg_parser.add_argument("--name", required=False, type=str, default="settings_experiment",
                            help="name of experiment")
    return vars(arg_parser.parse_args())


def command_for_settings_experiment():
    args = _parse_args()
    exposures = args['exposures']
    isos = args['isos']
    name = args['name']

    variant_parameters = ''

    for exposure in exposures:
        for iso in isos:
            variant_parameters += f'--variant exposure{exposure}_iso{iso} " -ss {exposure} -iso {iso}" '

    return f'run_experiment --name {name} {variant_parameters}'


def run_settings_experiment():
    command = command_for_settings_experiment()
    print(f'Running setting experiment: {command}')
    command_output = check_output(command, shell=True).decode("utf-8")
    return command_output


if __name__ == '__main__':
    run_settings_experiment()
