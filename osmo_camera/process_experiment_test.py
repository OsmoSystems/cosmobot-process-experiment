from unittest.mock import sentinel, Mock

import pytest

from . import process_experiment as module


@pytest.fixture
def mock_side_effects(mocker):
    mocker.patch.object(module, 'sync_from_s3').return_value = sentinel.raw_images_dir
    mocker.patch.object(module, 'raw')
    mocker.patch.object(module, '_open_first_image').return_value = sentinel.first_rgb_image
    mocker.patch.object(module, 'jupyter')
    mocker.patch.object(module, 'process_images').return_value = sentinel.image_summary_data
    mocker.patch.object(module, 'draw_ROIs_on_image').return_value = sentinel.rgb_image_with_ROI_definitions
    mocker.patch.object(module, '_save_summary_statistics_csv')


@pytest.fixture
def mock_prompt_for_ROI_selection(mocker):
    return mocker.patch.object(module, 'prompt_for_ROI_selection')


@pytest.fixture
def mock_generate_summary_images(mocker):
    return mocker.patch.object(module, 'generate_summary_images')


class TestProcessExperiment:
    def test_returns_image_summary_data_and_ROI_definitions(self, mock_side_effects):
        actual_image_summary_data, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.raspiraw_location,
            sentinel.local_sync_path,
            ROI_definitions=sentinel.ROI_definitions,
        )

        assert actual_image_summary_data == sentinel.image_summary_data
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_prompts_ROI_if_not_provided(self, mock_side_effects, mock_prompt_for_ROI_selection):
        mock_prompt_for_ROI_selection.return_value = sentinel.prompted_ROI_definitions

        actual_image_summary_data, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.raspiraw_location,
            sentinel.local_sync_path,
            ROI_definitions=None,
        )

        mock_prompt_for_ROI_selection.assert_called_with(sentinel.first_rgb_image)
        assert actual_ROI_definitions == sentinel.prompted_ROI_definitions

    def test_doesnt_prompt_ROI_if_provided(self, mock_side_effects, mock_prompt_for_ROI_selection):
        actual_image_summary_data, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.raspiraw_location,
            sentinel.local_sync_path,
            ROI_definitions=sentinel.ROI_definitions,
        )

        mock_prompt_for_ROI_selection.assert_not_called()
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_saves_summary_images_if_flagged(self, mock_side_effects, mock_generate_summary_images):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.raspiraw_location,
            sentinel.local_sync_path,
            ROI_definitions=sentinel.ROI_definitions,
            save_summary_images=True,
        )

        mock_generate_summary_images.assert_called_with(sentinel.raw_images_dir, sentinel.ROI_definitions)

    def test_doesnt_save_summary_images_if_not_flagged(self, mock_side_effects, mock_generate_summary_images):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.raspiraw_location,
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
