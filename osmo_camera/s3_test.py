import datetime
from unittest.mock import sentinel, Mock

import pandas as pd
import pytest

from . import s3 as module


@pytest.fixture
def mock_get_filenames_from_s3(mocker):
    # _get_filenames_from_s3 uses boto to interact with s3
    # (through _list_camera_sensor_experiments_s3_bucket_contents); use this fixture to mock it.
    return mocker.patch.object(module, '_get_filenames_from_s3')


@pytest.fixture
def mock_list_camera_sensor_experiments_s3_bucket_contents(mocker):
    # list_camera_sensor_experiments_s3_bucket_contents uses boto to interact with s3; use this fixture to mock it.
    # If you are trying to avoid side-effects at a high level, note that using this is redundant to using
    # mock_get_filenames_from_s3()
    return mocker.patch.object(module, 'list_camera_sensor_experiments_s3_bucket_contents')


class TestSyncFromS3:
    test_filenames = [
        '2018-01-02--00-00-00_ss_31000_ISO_100.jpeg',
        '2018-01-03--00-00-00_ss_31000_ISO_100.jpeg',
        '2018-01-04--00-00-00_ss_31000_ISO_100.jpeg'
    ]

    def test_sync_all_files(
        self,
        mocker,
        mock_get_filenames_from_s3
    ):
        mock_get_filenames_from_s3.return_value = self.test_filenames
        mock_download_s3_files = mocker.patch.object(module, '_download_s3_files')
        module.sync_from_s3(
            'experiment_directory',
            'local_sync_path',
        )

        mock_download_s3_files.assert_called_with(
            'experiment_directory',
            self.test_filenames,
            'local_sync_path/experiment_directory'
        )

    def test_sync_partial_directory(
        self,
        mocker,
        mock_get_filenames_from_s3
    ):
        mock_get_filenames_from_s3.return_value = self.test_filenames
        mock_download_s3_files = mocker.patch.object(module, '_download_s3_files')

        module.sync_from_s3(
            'experiment_directory',
            'local_sync_path',
            downsample_ratio=1,
            start_time=datetime.datetime(2018, 1, 1),
            end_time=datetime.datetime(2018, 1, 3),
        )

        mock_download_s3_files.assert_called_with(
            'experiment_directory',
            self.test_filenames[0:2],
            'local_sync_path/experiment_directory'
        )


class TestDownloadS3Files:
    def test_download_single_file(self, mocker):
        mock_get_contents_to_filename = Mock()
        mock_key = Mock(get_contents_to_filename=mock_get_contents_to_filename)

        mock_get_key = Mock()
        mock_get_key.return_value = mock_key
        mock_bucket = Mock(get_key=mock_get_key)

        module._download_s3_file(
            experiment_directory='my_experiment',
            file_name='image1.jpg',
            output_directory_path='local_sync_path',
            bucket=mock_bucket
        )

        mock_get_contents_to_filename.assert_called_once()

    def test_skips_download_if_exists(self, mocker):
        mocker.patch('os.path.isfile').return_value = True
        mock_get_contents_to_filename = Mock()
        mock_key = Mock(get_contents_to_filename=mock_get_contents_to_filename)

        mock_get_key = Mock()
        mock_get_key.return_value = mock_key
        mock_bucket = Mock(get_key=mock_get_key)

        module._download_s3_file(
            experiment_directory='my_experiment',
            file_name='image1.jpg',
            output_directory_path='local_sync_path',
            bucket=mock_bucket
        )

        assert mock_get_contents_to_filename.call_count == 0

    def test_calls_boto_get_file(self, mocker):
        mocker.patch.object(module, '_get_experiments_bucket').return_value = sentinel.bucket
        mock_download_file = mocker.patch.object(module, '_download_s3_file')

        module._download_s3_files(
            experiment_directory='my_experiment',
            file_names=[
                'image1.jpeg',
                'image2.jpeg',
            ],
            output_directory_path='local_sync_path'
        )

        assert mock_download_file.call_count == 2
        mock_download_file.assert_any_call(
            'my_experiment',
            'image1.jpeg',
            'local_sync_path',
            sentinel.bucket
        )
        mock_download_file.assert_any_call(
            'my_experiment',
            'image2.jpeg',
            'local_sync_path',
            sentinel.bucket
        )


class TestGetImagesInfo:
    def test_creates_appropriate_dataframe_ignoring_non_jpeg_files(self):
        image_filenames = [
            '2018-10-27--21-24-17_ss_31000_ISO_100.jpeg',
            '2018-10-27--21-24-23_ss_1_ISO_100.jpeg',
            'ignored_file.md'
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
            module._get_images_info(image_filenames),
            expected_images_info
        )

    def test_returns_empty_dataframe_if_no_files(self):
        images_df = module._get_images_info([])
        pd.testing.assert_frame_equal(
            images_df,
            pd.DataFrame(columns=module._IMAGES_INFO_COLUMNS)
        )


class TestGetFilenamesFromS3:
    experiment_directory = 'yyyy-mm-dd-experiment_name'

    def test_calls_list_with_correctly_appended_slash_on_experiment_directory(
        self, mock_list_camera_sensor_experiments_s3_bucket_contents
    ):
        mock_list_camera_sensor_experiments_s3_bucket_contents.return_value = []

        # Experiment directory has no trailing slash; the slash should be added by
        # list_camera_sensor_experiments_s3_bucket_contents.
        # If it's not added, we'll also get files from directories with longer names than the one we actually want
        module._get_filenames_from_s3('my_experiment')

        mock_list_camera_sensor_experiments_s3_bucket_contents.assert_called_once_with('my_experiment/')

    @pytest.mark.parametrize('bucket_contents, expected_filenames', [
        ([], []),
        (
            [
                f'{experiment_directory}/2018-10-27--21-24-17_ss_31000_ISO_100.jpeg',
                f'{experiment_directory}/experiment_metadata.yml'
            ],
            [
                '2018-10-27--21-24-17_ss_31000_ISO_100.jpeg',
                'experiment_metadata.yml',
            ]
        ),
    ])
    def test_returns_list_containing_all_filenames(
        self,
        bucket_contents,
        expected_filenames,
        mock_list_camera_sensor_experiments_s3_bucket_contents
    ):
        mock_list_camera_sensor_experiments_s3_bucket_contents.return_value = bucket_contents

        actual_filenames = module._get_filenames_from_s3(self.experiment_directory)

        assert actual_filenames == expected_filenames


class TestGetNonImageFilenames:
    def test_filters_to_non_image_filenames(self):
        assert module._get_non_image_filenames(['not an image', 'image.jpeg']) == ['not an image']


class TestGetCaptureGroups:
    def test_no_images__capture_groups_are_empty(self):
        variants = pd.Series()
        expected_capture_groups = pd.Series()

        pd.testing.assert_series_equal(
            module._get_capture_groups(variants),
            expected_capture_groups
        )

    def test_single_variant__capture_groups_are_just_a_linear_series(self):
        variants = pd.Series([
            'abc', 'abc', 'abc', 'abc', 'abc',
        ])
        expected_capture_groups = pd.Series([0, 1, 2, 3, 4])

        pd.testing.assert_series_equal(
            module._get_capture_groups(variants),
            expected_capture_groups
        )

    def test_two_variants__capture_groups_match_up(self):
        variants = pd.Series([
            'abc', 'd', 'abc', 'd', 'abc', 'd', 'abc', 'd', 'abc', 'd',
        ])
        expected_capture_groups = pd.Series(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        )

        pd.testing.assert_series_equal(
            module._get_capture_groups(variants),
            expected_capture_groups
        )

    def test_missing_variants_at_end__creates_small_capture_group_at_end(self):
        variants = pd.Series([
            'abc', 'd', 'abc', 'd', 'abc',
        ])
        expected_capture_groups = pd.Series(
            [0, 0, 1, 1, 2]
        )

        pd.testing.assert_series_equal(
            module._get_capture_groups(variants),
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
        mocker.patch.object(module, 'list_camera_sensor_experiments_s3_bucket_contents').return_value = [
            '2018-01-01--12-01-01_directory_1/',
            '2018-01-01--12-02-01_directory_2/',
        ]
        assert module.list_experiments() == ['2018-01-01--12-02-01_directory_2', '2018-01-01--12-01-01_directory_1']


UNORDERED_UNFILTERED_LIST_FOR_TESTS = [
    '20180902103709_temperature',
    '20180902103940_temperature',
    '2018-11-08--11-25-27-Pi4E82-test',
    '2018-11-08--11-26-00-Pi4E82-test',
    'should_be_filtered.jpng'
]


class TestFilterAndSortExperimentList:
    def test_returns_filtered_list_for_new_isodate_format(self):
        actual_filtered_list = module._experiment_list_by_isodate_format_date_desc(UNORDERED_UNFILTERED_LIST_FOR_TESTS)
        expected_filtered_list = [
            '2018-11-08--11-26-00-Pi4E82-test',
            '2018-11-08--11-25-27-Pi4E82-test'
        ]
        assert actual_filtered_list == expected_filtered_list
