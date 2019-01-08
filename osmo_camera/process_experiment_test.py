from unittest.mock import sentinel, Mock

import pandas as pd
import pytest

from . import process_experiment as module


@pytest.fixture
def mock_side_effects(mocker):
    mocker.patch.object(module, 'sync_from_s3').return_value = sentinel.raw_images_dir
    mocker.patch.object(module, '_get_first_image').return_value = sentinel.first_rgb_image
    mocker.patch.object(module, 'jupyter')
    mocker.patch.object(module, 'process_images').return_value = (sentinel.roi_summary_data, sentinel.image_diagnostics)
    mocker.patch.object(module, 'draw_ROIs_on_image').return_value = sentinel.rgb_image_with_ROI_definitions
    mocker.patch.object(module, '_save_summary_statistics_csv')
    mocker.patch.object(module, 'get_rgb_images_by_filepath').return_value = sentinel.rgb_images_by_filepath


@pytest.fixture
def mock_prompt_for_ROI_selection(mocker):
    return mocker.patch.object(module, 'prompt_for_ROI_selection')


@pytest.fixture
def mock_generate_summary_images(mocker):
    return mocker.patch.object(module, 'generate_summary_images')


@pytest.fixture
def mock_os_path_join(mocker):
    return mocker.patch('os.path.join')


class TestProcessExperiment:
    def test_returns_image_summary_dataframes_and_ROI_definitions(self, mock_side_effects):
        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            ROI_definitions=sentinel.ROI_definitions,
        )

        assert actual_roi_summary_data == sentinel.roi_summary_data
        assert actual_image_diagnostics == sentinel.image_diagnostics
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_prompts_ROI_if_not_provided(self, mock_side_effects, mock_prompt_for_ROI_selection):
        mock_prompt_for_ROI_selection.return_value = sentinel.prompted_ROI_definitions

        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            ROI_definitions=None,
        )

        mock_prompt_for_ROI_selection.assert_called_with(sentinel.first_rgb_image)
        assert actual_ROI_definitions == sentinel.prompted_ROI_definitions

    def test_doesnt_prompt_ROI_if_provided(self, mock_side_effects, mock_prompt_for_ROI_selection):
        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            ROI_definitions=sentinel.ROI_definitions,
        )

        mock_prompt_for_ROI_selection.assert_not_called()
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_saves_summary_images_if_flagged(self, mock_side_effects, mock_generate_summary_images):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            ROI_definitions=sentinel.ROI_definitions,
            save_summary_images=True,
        )

        mock_generate_summary_images.assert_called_with(
            sentinel.rgb_images_by_filepath,
            sentinel.ROI_definitions,
            sentinel.raw_images_dir
        )

    def test_doesnt_save_summary_images_if_not_flagged(self, mock_side_effects, mock_generate_summary_images):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            ROI_definitions=sentinel.ROI_definitions,
        )

        mock_generate_summary_images.assert_not_called()


class TestSaveSummaryStatisticsCsv:
    def test_names_csv_with_current_iso_ish_datetime(self, mocker):
        mocker.patch.object(module, 'iso_datetime_for_filename').return_value = '<iso_ish_datetime>'

        mock_to_csv = Mock()
        mock_image_summary_data = Mock(to_csv=mock_to_csv)
        module._save_summary_statistics_csv('20180101-120101_experiment_dir', mock_image_summary_data)

        expected_csv_name = '20180101-120101_experiment_dir - summary statistics (generated <iso_ish_datetime>).csv'

        mock_to_csv.assert_called_with(expected_csv_name, index=False)


def test_get_first_image():
    mock_rgb_images_by_filepath = pd.Series({
        '2017-01-01-image': sentinel.image_2,
        '2018-01-01-image': sentinel.image_3,
        '2016-01-01-image': sentinel.image_1,
    })
    actual = module._get_first_image(mock_rgb_images_by_filepath)

    assert actual == sentinel.image_1


def test_get_rgb_images_by_filepath(mocker, mock_os_path_join):
    mock_get_files_with_extension = mocker.patch.object(module, 'get_files_with_extension')
    mock_get_files_with_extension.return_value = ['filepath1.jpeg', 'filepath2.jpeg']

    mock_raw_open_as_rgb = mocker.patch.object(module.raw.open, 'as_rgb')
    mock_raw_open_as_rgb.side_effect = lambda filepath: f'opened_{filepath}'

    actual = module.get_rgb_images_by_filepath(sentinel.local_sync_directory, sentinel.experiment_directory)
    expected = pd.Series({
        'filepath1.jpeg': 'opened_filepath1.jpeg',
        'filepath2.jpeg': 'opened_filepath2.jpeg'
    })

    pd.testing.assert_series_equal(actual, expected)
