import datetime
# import tmpdir


VALID_CONFIGURATION = {
    'command': 'experiment.py --interval 10 --name exp1 --variant variant1  -ss 100 -iso 100',
    'git_hash': 'fac9e3099920cadb0d856950baa6bacaf801e52a',
    'hostname': 'pi-cam-2222',
    'interval': 2,
    'name': 'exp1',
    'start_date': datetime.datetime(2018, 10, 4, 6, 12, 49, 336313),
    'duration': 10,
    'experiment_output_folder': '../output/20181004061249_exp1',
    'variants': [{
        'name': 'variant1',
        'capture_params': ' -ss 100 -iso 100',
        'output_folder': '../output/20181004061249_exp1/variant1',
        'metadata': {}
    }]
}

def test_perform_experiment():
    '''TODO:'''
    assert True
    # create temp
