from unittest.mock import sentinel
import pytest

from . import convert as module


class TestFileToDng:
    def test_constructs_and_calls_raspiraw_command(self, mocker):
        mock_call = mocker.patch('subprocess.check_call')
        mock_isfile = mocker.patch('os.path.isfile')
        mock_isfile.return_value = False

        module._file_to_dng('/raspiraw/location', raw_image_path='raw_image.jpeg')

        expected_command = '/raspiraw/location/raspiraw/raspi_dng_sony "raw_image.jpeg" "raw_image.dng"'
        mock_call.assert_called_with([expected_command], shell=True)

    def test_does_not_convert_if_already_exists(self, mocker):
        mock_call = mocker.patch('subprocess.check_call')
        mock_isfile = mocker.patch('os.path.isfile')
        mock_isfile.return_value = True

        module._file_to_dng('/raspiraw/location', raw_image_path='raw_image.jpeg')

        mock_call.assert_not_called()


class TestToDng:
    @pytest.mark.parametrize("name,raw_image_path,raw_images_dir", [
        ('missing parameters', None, None),
        ('too many parameters', sentinel.not_none, sentinel.not_none),
    ])
    def test_raises_if_incorrect_parameters(self, name, raw_image_path, raw_images_dir):
        with pytest.raises(Exception):
            module.to_dng(sentinel.mock_raspiraw_location, raw_image_path, raw_images_dir)

    def test_calls_directory_to_dng_if_raw_images_dir(self, mocker):
        mock_directory_to_dng = mocker.patch.object(module, '_directory_to_dng')

        module.to_dng(sentinel.mock_raspiraw_location, raw_image_path=None, raw_images_dir=sentinel.not_none)

        mock_directory_to_dng.assert_called()

    def test_calls_file_to_dng_if_raw_image_path(self, mocker):
        mock_file_to_dng = mocker.patch.object(module, '_file_to_dng')

        module.to_dng(sentinel.mock_raspiraw_location, raw_image_path=sentinel.not_none, raw_images_dir=None)

        mock_file_to_dng.assert_called()
