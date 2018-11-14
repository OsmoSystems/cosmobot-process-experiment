import datetime

import pandas as pd
import pytest

from . import s3 as module


@pytest.fixture
def mock_check_call(mocker):
    mock_check_call = mocker.patch.object(module, 'check_call')
    mock_check_call.return_value = None

    return mock_check_call


@pytest.fixture
def mock_get_images_info(mocker):
    # _get_images_info uses boto to interact with s3 (through _list_camera_sensor_experiments_s3_bucket_contents);
    # use this fixture to mock it.
    return mocker.patch.object(module, '_get_images_info')


@pytest.fixture
def mock_list_camera_sensor_experiments_s3_bucket_contents(mocker):
    # _list_camera_sensor_experiments_s3_bucket_contents uses boto to interact with s3; use this fixture to mock it.
    # If you are trying to avoid side-effects at a high level, note that using this is redundant to using
    # mock_get_images_info()
    return mocker.patch.object(module, '_list_camera_sensor_experiments_s3_bucket_contents')


class TestSyncFromS3:
    def test_sync_from_s3_results_in_call_to_download_file(
        self,
        mocker,
        mock_check_call,
        mock_get_images_info
    ):
        ''' the method under test just calls a lot of sub-functions, so this is a smoke test only. '''
        mock_get_images_info.return_value = pd.DataFrame([
            {
                'Timestamp': datetime.datetime(2018, 1, 2),
                'capture_group': 0,
                'variant': 'some_variant',
                'filename': mocker.sentinel.filename
            }
        ])
        module.sync_from_s3(
            'experiment_directory',
            'local_sync_path',
            downsample_ratio=1,
            start_time=datetime.datetime(2018, 1, 1),
            end_time=datetime.datetime(2018, 1, 3),
        )

        # Should result in exactly one call to s3 do download the image
        mock_check_call.assert_called_once()

        # triple indexing: list of calls -> list of call args -> check_call argument 0 is itself a list.
        actual_s3_command = mock_check_call.call_args[0][0][0]
        assert 'aws s3 sync' in actual_s3_command
        assert str(mocker.sentinel.filename) in actual_s3_command


class TestDownloadS3Files:
    def test_calls_s3_sync_command(self, mock_check_call):
        module._download_s3_files(
            experiment_directory='my_experiment',
            file_names=[
                'image1.jpeg',
                'image2.jpeg',
            ],
            output_directory='local_sync_path'
        )

        expected_command = (
            'aws s3 sync s3://camera-sensor-experiments/my_experiment local_sync_path '
            '--exclude "*" '
            '--include "image1.jpeg" --include "image2.jpeg"'
        )

        mock_check_call.assert_called_with([expected_command], shell=True)


class TestGetImagesInfo:
    def test_calls_list_with_correctly_appended_slash_on_experiment_directory(
        self, mock_list_camera_sensor_experiments_s3_bucket_contents
    ):
        mock_list_camera_sensor_experiments_s3_bucket_contents.return_value = []

        # Experiment directory has no trailing slash; the slash should be added by
        # _list_camera_sensor_experiments_s3_bucket_contents.
        # If it's not added, we'll also get files from directories with longer names than the one we actually want
        module._get_images_info('my_experiment')

        mock_list_camera_sensor_experiments_s3_bucket_contents.assert_called_once_with('my_experiment/')

    def test_creates_appropriate_dataframe(self, mock_list_camera_sensor_experiments_s3_bucket_contents):
        experiment_directory = 'yyyy-mm-dd-experiment_name'
        mock_list_camera_sensor_experiments_s3_bucket_contents.return_value = [
            f'{experiment_directory}/2018-10-27--21-24-17_ss_31000_ISO_100.jpeg',
            f'{experiment_directory}/2018-10-27--21-24-23_ss_1_ISO_100.jpeg',
            f'{experiment_directory}/experiment_metadata.yml',
        ]

        expected_images_info = pd.DataFrame([
            {
                'Timestamp': datetime.datetime(2018, 10, 27, 21, 24, 17),
                'variant': '_ss_31000_ISO_100',
                'filename': '2018-10-27--21-24-17_ss_31000_ISO_100.jpeg',
                'capture_group': 0,
            },
            {
                'Timestamp': datetime.datetime(2018, 10, 27, 21, 24, 23),
                'variant': '_ss_1_ISO_100',
                'filename': '2018-10-27--21-24-23_ss_1_ISO_100.jpeg',
                'capture_group': 0,
            }
        ], columns=module._IMAGES_INFO_COLUMNS)

        pd.testing.assert_frame_equal(
            module._get_images_info(experiment_directory),
            expected_images_info
        )

    def test_returns_empty_dataframe_if_no_files(self, mocker, mock_list_camera_sensor_experiments_s3_bucket_contents):
        mock_list_camera_sensor_experiments_s3_bucket_contents.return_value = []

        pd.testing.assert_frame_equal(
            module._get_images_info(mocker.sentinel.experiment_directory),
            pd.DataFrame(columns=module._IMAGES_INFO_COLUMNS)
        )


class TestGetCaptureGroups:
    def test_single_variant__capture_groups_are_just_a_linear_series(self):
        images_info = pd.Series([
            'abc', 'abc', 'abc', 'abc', 'abc',
        ])
        expected_capture_groups = pd.Series([0, 1, 2, 3, 4])

        pd.testing.assert_series_equal(
            module._get_capture_groups(images_info),
            expected_capture_groups
        )

    def test_two_variants__capture_groups_match_up(self):
        images_info = pd.Series([
            'abc', 'd', 'abc', 'd', 'abc', 'd', 'abc', 'd', 'abc', 'd',
        ])
        expected_capture_groups = pd.Series(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        )

        pd.testing.assert_series_equal(
            module._get_capture_groups(images_info),
            expected_capture_groups
        )

    def test_missing_variants_at_end__creates_small_capture_group_at_end(self):
        images_info = pd.Series([
            'abc', 'd', 'abc', 'd', 'abc',
        ])
        expected_capture_groups = pd.Series(
            [0, 0, 1, 1, 2]
        )

        pd.testing.assert_series_equal(
            module._get_capture_groups(images_info),
            expected_capture_groups
        )


class TestDownsample:
    def test_base_case(self):
        images_info = pd.DataFrame([
            {'capture_group': 0, 'variant': 'abc'},
            {'capture_group': 0, 'variant': 'def'},
            {'capture_group': 1, 'variant': 'abc'},
            {'capture_group': 1, 'variant': 'def'},
        ])
        pd.testing.assert_frame_equal(
            module._downsample(images_info, 1),
            images_info
        )

    @pytest.mark.parametrize("name, ratio, expected_capture_groups", [
        ('ratio 2 provides half', 2, [0, 2, 4]),
        ('ratio 3 provides 1/3', 3, [0, 3]),
        ('ratio > groups returns first group', 30, [0]),
    ])
    def test_downsampling(self, name, ratio, expected_capture_groups):
        images_info = pd.DataFrame([
            {'capture_group': 0, 'variant': 'abc'},
            {'capture_group': 0, 'variant': 'def'},
            {'capture_group': 1, 'variant': 'abc'},
            {'capture_group': 1, 'variant': 'def'},
            {'capture_group': 2, 'variant': 'abc'},
            {'capture_group': 2, 'variant': 'def'},
            {'capture_group': 3, 'variant': 'abc'},
            {'capture_group': 3, 'variant': 'def'},
            {'capture_group': 4, 'variant': 'abc'},
            {'capture_group': 4, 'variant': 'def'},
        ])
        expected_downsampled_df = images_info[images_info['capture_group'].isin(expected_capture_groups)]

        actual_downsampled_df = module._downsample(images_info, ratio)

        pd.testing.assert_frame_equal(actual_downsampled_df, expected_downsampled_df)


class TestFilterToTimeRange:
    @pytest.mark.parametrize("name, start_time, end_time, expected_indices", [
        ('leaves everything alone if no start or end time provided', None, None, [0, 1, 2, 3, 4]),
        ('filter start only, inclusive', datetime.datetime(2018, 2, 1), None, [1, 2, 3, 4]),
        ('filter end only, inclusive', None, datetime.datetime(2018, 3, 1), [0, 1, 2]),
        ('filter both', datetime.datetime(2018, 2, 15), datetime.datetime(2018, 4, 15), [2, 3]),
    ])
    def test_filtering(self, name, start_time, end_time, expected_indices):
        images_info = pd.DataFrame([
            {'Timestamp': datetime.datetime(2018, 1, 1)},
            {'Timestamp': datetime.datetime(2018, 2, 1)},
            {'Timestamp': datetime.datetime(2018, 3, 1)},
            {'Timestamp': datetime.datetime(2018, 4, 1)},
            {'Timestamp': datetime.datetime(2018, 5, 1)},
        ], index=[0, 1, 2, 3, 4])

        filtered_df = module._filter_to_time_range(
            images_info,
            start_time,
            end_time
        )
        assert list(filtered_df.index.values) == expected_indices


class TestListExperiments:
    def test_returns_cleaned_sorted_directories(self, mocker):
        mocker.patch.object(module, '_list_camera_sensor_experiments_s3_bucket_contents').return_value = [
            '2018-01-01--12-01-01_directory_1/',
            '2018-01-01--12-02-01_directory_2/',
        ]
        assert module.list_experiments() == ['2018-01-01--12-02-01_directory_2', '2018-01-01--12-01-01_directory_1']


class TestAlterExperimentList:

    unordered_unfiltered_list_for_tests = [
        '20180902103709_temperature', '20180902103940_temperature',
        '2018-11-08--11-25-27-Pi4E82-test', '2018-11-08--11-26-00-Pi4E82-test'
    ]

    def test_returns_ordered_list(self):
        actual_ordered_list = module.order_experiment_list_by_isodate_formats(self.unordered_unfiltered_list_for_tests)
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
