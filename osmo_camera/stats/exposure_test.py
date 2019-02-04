import pytest
import warnings
import numpy as np

from . import exposure as module

rgb_image = np.array([
    [[0.0, 0.1, 0.2], [0.1, 0.2, 0.0]],
    [[0.0, 0.1, 0.999], [0.0, 0.999, 0.999]]
])

rgb_images_by_filename = {
    '1.jpeg': rgb_image
}


class TestGenerateStatistics():

    def test_overexposed_percent(self):
        stats = module._generate_statistics(rgb_image)
        actual_overexposed_percent = stats['overexposed_percent']
        expected_overexposed_percent = 0.25
        assert actual_overexposed_percent == expected_overexposed_percent

    def test_underexposed_percent(self):
        stats = module._generate_statistics(rgb_image)
        actual_underexposed_percent = stats['underexposed_percent']
        expected_underexposed_percent = 0.3333333333333333
        assert actual_underexposed_percent == expected_underexposed_percent

    @pytest.mark.parametrize("color_channel_key, expected_invalid_exposure_percent", [
        ("overexposed_percent_r", 0.0),
        ("overexposed_percent_g", 0.25),
        ("overexposed_percent_b", 0.5),
        ("underexposed_percent_r", 0.75),
        ("underexposed_percent_g", 0.0),
        ("underexposed_percent_b", 0.25)
    ])
    def test_exposure_percent_by_color_channel(self, color_channel_key, expected_invalid_exposure_percent):
        stats = module._generate_statistics(rgb_image)
        actual_invalid_exposure_percent = stats[color_channel_key]
        assert actual_invalid_exposure_percent == expected_invalid_exposure_percent


class TestExposureWarnings():

    def test_warnings_raised(self, mocker):
        mock_warn = mocker.patch.object(warnings, 'warn')
        mock_warn.return_value = None

        module.warn_if_exposure_out_of_range(rgb_images_by_filename)
        assert mock_warn.call_count == 2
