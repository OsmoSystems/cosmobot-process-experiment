from unittest.mock import sentinel

import pytest

from . import s3 as module


@pytest.fixture
def mock_path_join(mocker):
    mock_path_join = mocker.patch('os.path.join')
    mock_path_join.return_value = 'sync_dir_location'

    return mock_path_join


@pytest.fixture
def mock_check_call(mocker):
    mock_check_call = mocker.patch.object(module, 'check_call')
    mock_check_call.return_value = None

    return mock_check_call


@pytest.fixture
def mock_os(mocker):
    mock_gettempdir = mocker.patch('tempfile.gettempdir')
    mock_gettempdir.return_value = sentinel.tempdir

    return mocker.patch('platform.system')


class TestSyncFromS3:
    def test_returns_sync_dir_location(self, mock_check_call, mock_path_join):
        actual = module.sync_from_s3('experiment_directory_name', local_sync_dir='local_sync_dir')

        assert actual == 'sync_dir_location'

    def test_calls_s3_sync_command(self, mock_check_call, mock_path_join):
        module.sync_from_s3('experiment_directory_name', local_sync_dir='local_sync_dir')

        expected_command = f'aws s3 sync s3://camera-sensor-experiments/experiment_directory_name sync_dir_location'

        mock_check_call.assert_called_with([expected_command], shell=True)

    @pytest.mark.parametrize("name,local_sync_dir,os_name,expected_sync_dir", [
        ('syncs to local_sync_dir if provided', 'local_sync_dir', 'Not Darwin', 'local_sync_dir'),
        ('syncs to gettempdir if not mac ', None, 'Not Darwin', sentinel.tempdir),
        ('syncs to /tmp if mac ', None, 'Darwin', '/tmp'),
    ])
    def test_local_sync_dir_conditions(
        self,
        name,
        local_sync_dir,
        os_name,
        expected_sync_dir,
        mock_path_join,
        mock_os,
        mock_check_call
    ):
        mock_os.return_value = os_name
        module.sync_from_s3('experiment_directory_name', local_sync_dir)

        mock_path_join.assert_called_with(expected_sync_dir, 'experiment_directory_name')


class TestAlterExperimentList:

    unordered_unfiltered_list_for_tests = [
        '20180902103709_temperature', '20180902103940_temperature',
        '2018-11-08--11-25-27-Pi4E82-test', '2018-11-08--11-26-00-Pi4E82-test'
    ]

    def test_returns_ordered_list(self):
        actual_ordered_list = module.order_experiment_list(self.unordered_unfiltered_list_for_tests)
        expected_ordered_list = [
            '2018-11-08--11-26-00-Pi4E82-test', '2018-11-08--11-25-27-Pi4E82-test',
            '20180902103940_temperature', '20180902103709_temperature'
        ]
        assert actual_ordered_list == expected_ordered_list

    def test_returns_filtered_list_for_new_isodate_format(self):
        actual_filtered_list = module.filter_and_reverse_experiment_list(
            self.unordered_unfiltered_list_for_tests,
            r'^\d{4}-\d\d-\d\d.'
        )
        expected_filtered_list = ['2018-11-08--11-26-00-Pi4E82-test', '2018-11-08--11-25-27-Pi4E82-test']
        assert actual_filtered_list == expected_filtered_list

    def test_returns_filtered_list_for_old_isodate_format(self):
        actual_filtered_list = module.filter_and_reverse_experiment_list(
            self.unordered_unfiltered_list_for_tests,
            r'^\d{8}.'
        )
        expected_filtered_list = ['20180902103940_temperature', '20180902103709_temperature']
        assert actual_filtered_list == expected_filtered_list
