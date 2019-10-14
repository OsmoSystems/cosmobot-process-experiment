import warnings
from unittest.mock import sentinel

import pandas as pd
import pytest
import os

from . import process_experiment as module

# Needed for `testdir` fixture
pytest_plugins = "pytester"


@pytest.fixture
def mock_side_effects(mocker):
    mocker.patch.object(module, "sync_from_s3").return_value = sentinel.raw_images_dir
    mocker.patch.object(
        module, "_open_first_image"
    ).return_value = sentinel.first_rgb_image
    mocker.patch.object(module, "jupyter")
    mocker.patch.object(module, "process_image").return_value = (
        pd.DataFrame([{"mock ROI statistic": sentinel.roi_summary_statistic}]),
        pd.Series({"mock image diagnostic": sentinel.image_diagnostic}),
    )
    mocker.patch.object(
        module.rgb.annotate, "draw_ROIs_on_image"
    ).return_value = sentinel.rgb_image_with_ROI_definitions
    mocker.patch.object(
        module.raw.open, "as_rgb"
    ).return_value = sentinel.opened_image_filepath
    mocker.patch.object(module, "get_raw_image_paths_for_experiment").return_value = [
        sentinel.image_filepath
    ]


@pytest.fixture
def mock_prompt_for_ROI_selection(mocker):
    return mocker.patch.object(module, "prompt_for_ROI_selection")


@pytest.fixture
def mock_generate_summary_images(mocker):
    return mocker.patch.object(module, "generate_summary_images")


@pytest.fixture
def mock_os_path_join(mocker):
    return mocker.patch("os.path.join")


def _process_image_stub_with_warning(**kwargs):
    warnings.warn("Diagnostic warning!!")
    return (
        pd.DataFrame([{"mock ROI statistic": sentinel.roi_summary_statistic}]),
        pd.Series({"mock image diagnostic": sentinel.image_diagnostic}),
    )


class TestProcessExperiment:
    def test_returns_image_summary_dataframes_and_ROI_definitions(
        self, mock_side_effects
    ):
        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=sentinel.ROI_definitions,
        )

        expected_roi_summary_data = pd.DataFrame(
            [{"mock ROI statistic": sentinel.roi_summary_statistic}]
        )
        expected_image_diagnostics = pd.DataFrame(
            [{"mock image diagnostic": sentinel.image_diagnostic}]
        )

        pd.testing.assert_frame_equal(
            actual_roi_summary_data, expected_roi_summary_data
        )
        pd.testing.assert_frame_equal(
            actual_image_diagnostics, expected_image_diagnostics
        )
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_prompts_ROI_if_not_provided(
        self, mock_side_effects, mock_prompt_for_ROI_selection
    ):
        mock_prompt_for_ROI_selection.return_value = sentinel.prompted_ROI_definitions

        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=None,
        )

        mock_prompt_for_ROI_selection.assert_called_with(sentinel.first_rgb_image)
        assert actual_ROI_definitions == sentinel.prompted_ROI_definitions

    def test_doesnt_prompt_ROI_if_provided(
        self, mock_side_effects, mock_prompt_for_ROI_selection
    ):
        actual_roi_summary_data, actual_image_diagnostics, actual_ROI_definitions = module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=sentinel.ROI_definitions,
        )

        mock_prompt_for_ROI_selection.assert_not_called()
        assert actual_ROI_definitions == sentinel.ROI_definitions

    def test_saves_summary_images_if_flagged(
        self, mocker, mock_side_effects, mock_generate_summary_images
    ):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=sentinel.ROI_definitions,
            save_summary_images=True,
        )

        mock_generate_summary_images.assert_called_with(
            [sentinel.image_filepath], sentinel.ROI_definitions, sentinel.raw_images_dir
        )

    def test_doesnt_save_summary_images_if_not_flagged(
        self, mock_side_effects, mock_generate_summary_images
    ):
        module.process_experiment(
            sentinel.experiment_dir,
            sentinel.local_sync_path,
            flat_field_filepath=sentinel.flat_field_filepath,
            ROI_definitions=sentinel.ROI_definitions,
        )

        mock_generate_summary_images.assert_not_called()


class TestProcessImages:
    def test_matching_diagnostic_warnings_raised_only_once(
        self, mocker, mock_side_effects
    ):
        mock_process_image = mocker.patch.object(module, "process_image")
        mock_process_image.side_effect = _process_image_stub_with_warning

        with warnings.catch_warnings(record=True) as _warnings:
            module._process_images(
                raw_image_paths=[
                    sentinel.image_filepath_one,
                    sentinel.image_filepath_two,
                ],
                raw_images_dir=sentinel.experiment_dir,
                ROI_definitions=sentinel.ROI_definitions,
                flat_field_filepath_or_none=sentinel.flat_field_filepath,
                save_ROIs=sentinel.save_ROIs,
                save_dark_frame_corrected_images=sentinel.save_dark_frame_corrected_images,
                save_flat_field_corrected_images=sentinel.save_flat_field_corrected_images,
            )

        # meta-test to make sure we're actually getting two images processed here
        assert mock_process_image.call_count == 2

        # Each call to process_image will attempt to raise a warning, but the warnings system
        # should handle it such that only the first of the identical warnings is raised
        # mypy thinks this could be None but it's not
        assert len(_warnings) == 1  # type: ignore

    def test_matching_diagnostic_warnings_once_per_process_experiment_run(
        self, mocker, mock_side_effects
    ):
        mocker.patch.object(
            module, "process_image"
        ).side_effect = _process_image_stub_with_warning

        with warnings.catch_warnings(record=True) as _warnings:
            module.process_experiment(
                sentinel.experiment_dir,
                sentinel.local_sync_path,
                flat_field_filepath=sentinel.flat_field_filepath,
                ROI_definitions=sentinel.ROI_definitions,
            )
            module.process_experiment(
                sentinel.experiment_dir,
                sentinel.local_sync_path,
                flat_field_filepath=sentinel.flat_field_filepath,
                ROI_definitions=sentinel.ROI_definitions,
            )

        # mypy thinks this could be None but it's not
        assert len(_warnings) == 2  # type: ignore


class TestSaveSummaryStatisticsCsv:
    def test_names_csv_with_current_iso_ish_datetime(self, testdir, mocker):
        mocker.patch.object(
            module, "iso_datetime_for_filename"
        ).return_value = "<iso_ish_datetime>"

        mock_image_summary_data = pd.DataFrame(
            [{"mock ROI statistic": sentinel.roi_summary_statistic}]
        )
        module.save_summary_statistics_csv(
            "20180101-120101_experiment_dir", mock_image_summary_data
        )
        expected_csv_name = "20180101-120101_experiment_dir - summary statistics (generated <iso_ish_datetime>).csv"

        assert os.path.isfile(expected_csv_name)

        testdir.finalize()


def test_open_first_image(mocker):
    first_image_filename = "2016-01-01-image"
    mock_rgb_image_paths = pd.Series(
        ["2017-01-01-image", "2018-01-01-image", first_image_filename]
    )
    mock_raw_open_as_rgb = mocker.patch.object(module.raw.open, "as_rgb")
    mock_raw_open_as_rgb.side_effect = lambda filepath: f"opened_{filepath}"

    actual = module._open_first_image(mock_rgb_image_paths)

    assert actual == f"opened_{first_image_filename}"


def test_get_raw_image_paths_for_experiment(mocker, mock_os_path_join):
    mock_get_files_with_extension = mocker.patch.object(
        module, "get_files_with_extension"
    )
    mock_get_files_with_extension.return_value = ["filepath1.jpeg", "filepath2.jpeg"]

    actual = module.get_raw_image_paths_for_experiment(
        sentinel.local_sync_directory, sentinel.experiment_directory
    )
    expected = pd.Series(["filepath1.jpeg", "filepath2.jpeg"])

    pd.testing.assert_series_equal(actual, expected)


class TestStackSeries:
    def test_stack_dataframes_stacks_appropriately_resetting_index(self):
        df1 = pd.DataFrame([{"a": "b1", "c": "d1"}])
        df2 = pd.DataFrame([{"a": "b2", "c": "d2"}, {"a": "b2", "c": "d3"}])

        actual = module._stack_dataframes([df1, df2])

        expected = pd.DataFrame(
            [{"a": "b1", "c": "d1"}, {"a": "b2", "c": "d2"}, {"a": "b2", "c": "d3"}],
            index=[0, 1, 2],
        )

        pd.testing.assert_frame_equal(actual, expected)


class TestStackSerieses:
    def test_stack_serieses_stacks_appropriately_applying_names_to_index(self):
        ser1 = pd.Series({"a": "b1", "c": "d1"})
        ser1.name = "row 1"
        ser2 = pd.Series({"a": "b2", "c": "d2"})
        ser2.name = "row 2"

        actual = module._stack_serieses([ser1, ser2])

        expected = pd.DataFrame(
            [{"a": "b1", "c": "d1"}, {"a": "b2", "c": "d2"}], index=["row 1", "row 2"]
        )

        pd.testing.assert_frame_equal(actual, expected)
