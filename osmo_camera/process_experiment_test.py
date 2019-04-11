from unittest.mock import sentinel, Mock

import pandas as pd
import pytest

from . import process_experiment as module


@pytest.fixture
def mock_side_effects(mocker):
    mocker.patch.object(module, 'sync_from_s3').return_value = sentinel.raw_images_dir
    mocker.patch.object(module, '_open_first_image').return_value = sentinel.first_rgb_image
    mocker.patch.object(module, 'jupyter')
    mocker.patch.object(module, 'process_images').return_value = (
        pd.DataFrame([{'mock ROI statistic': sentinel.roi_summary_statistic}]),
        pd.DataFrame([{'mock image diagnostic': sentinel.image_diagnostic}]),
    )
    mocker.patch.object(module, 'draw_ROIs_on_image').return_value = sentinel.rgb_image_with_ROI_definitions
    mocker.patch.object(module, '_save_summary_statistics_csv')
    mocker.patch.object(module.raw.open, 'as_rgb').return_value = sentinel.opened_image_filepath
    mocker.patch.object(module, 'get_raw_image_paths_for_experiment').return_value = [sentinel.image_filepath]


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
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=sentinel.ROI_definitions,
        )

        assert isinstance(actual_roi_summary_data, pd.DataFrame)
        assert actual_roi_summary_data['mock ROI statistic'][0] == sentinel.roi_summary_statistic
        assert isinstance(actual_image_diagnostics, pd.DataFrame)
        assert actual_image_diagnostics['mock image diagnostic'][0] == sentinel.image_diagnostic
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_prompts_ROI_if_not_provided(self, mock_side_effects, mock_prompt_for_ROI_selection):
        mock_prompt_for_ROI_selection.return_value = sentinel.prompted_ROI_definitions

        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=None,
        )

        mock_prompt_for_ROI_selection.assert_called_with(sentinel.first_rgb_image)
        assert actual_ROI_definitions == sentinel.prompted_ROI_definitions

    def test_doesnt_prompt_ROI_if_provided(self, mock_side_effects, mock_prompt_for_ROI_selection):
        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=sentinel.ROI_definitions,
        )

        mock_prompt_for_ROI_selection.assert_not_called()
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_saves_summary_images_if_flagged(self, mocker, mock_side_effects, mock_generate_summary_images):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=sentinel.ROI_definitions,
            save_summary_images=True,
        )

        expected_call_args = (
            pd.Series({sentinel.image_filepath: sentinel.opened_image_filepath}),
            sentinel.ROI_definitions,
            sentinel.raw_images_dir
        )

        # It would be nice to just use assert_called_with(*expected_call_args) here but one of the call args is a
        # Series. Serieses don't like equality testing, so we have to go through a somewhat protracted process instead:
        mock_generate_summary_images.assert_called()
        # Grab the first call; then grab positional args (not keyword args)
        call_args = mock_generate_summary_images.call_args_list[0][0]
        pd.testing.assert_series_equal(
            call_args[0], expected_call_args[0]
        )
        assert call_args[1:] == expected_call_args[1:]

    def test_doesnt_save_summary_images_if_not_flagged(self, mock_side_effects, mock_generate_summary_images):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
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


def test_open_first_image(mocker):
    first_image_filename = '2016-01-01-image'
    mock_rgb_image_paths = pd.Series([
        '2017-01-01-image',
        '2018-01-01-image',
        first_image_filename,
    ])
    mock_raw_open_as_rgb = mocker.patch.object(module.raw.open, 'as_rgb')
    mock_raw_open_as_rgb.side_effect = lambda filepath: f'opened_{filepath}'

    actual = module._open_first_image(mock_rgb_image_paths)

    assert actual == f'opened_{first_image_filename}'


def test_get_raw_image_paths_for_experiment(mocker, mock_os_path_join):
    mock_get_files_with_extension = mocker.patch.object(module, 'get_files_with_extension')
    mock_get_files_with_extension.return_value = ['filepath1.jpeg', 'filepath2.jpeg']

    actual = module.get_raw_image_paths_for_experiment(sentinel.local_sync_directory, sentinel.experiment_directory)
    expected = pd.Series([
        'filepath1.jpeg',
        'filepath2.jpeg',
    ])

    pd.testing.assert_series_equal(actual, expected)
