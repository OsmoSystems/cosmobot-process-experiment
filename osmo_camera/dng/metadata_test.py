from datetime import datetime
from unittest.mock import Mock, sentinel

import pytest

from . import metadata as module


class TestReadExifTags:
    def test_parses_tag_codes_as_names(self, mocker):
        # Keys are actual EXIF tag codes
        mock_exif_codes_to_values = {
            36867: '2018:09:28 20:29:59',
            33434: (599985, 1000000),
            34855: 1,
        }
        mock_PIL_image = Mock(_getexif=Mock(return_value=mock_exif_codes_to_values))
        mocker.patch('PIL.Image.open').return_value = mock_PIL_image

        actual = module._read_exif_tags(sentinel.image_path)

        expected = {
            'DateTimeOriginal': '2018:09:28 20:29:59',
            'ExposureTime': (599985, 1000000),
            'ISOSpeedRatings': 1,
        }

        assert actual == expected


class TestParseDatTimeOriginal:
    def test_parses_ISO_string(self):
        actual = module._parse_date_time_original({'DateTimeOriginal': '2018:09:28 20:29:59'})
        expected = datetime(2018, 9, 28, 20, 29, 59)

        assert actual == expected

    def test_raises_on_missing_tag(self):
        with pytest.raises(KeyError):
            module._parse_date_time_original({})


class TestParseExposureTime:
    def test_parses_exposure_fraction_correctly(self):
        actual = module._parse_exposure_time({'ExposureTime': (599985, 1000000)})
        expected = 599985 / 1000000

        assert actual == expected

    def test_raises_on_missing_tag(self):
        with pytest.raises(KeyError):
            module._parse_exposure_time({})


class TestParseISO:
    def test_parses_iso_correctly(self):
        actual = module._parse_iso({'ISOSpeedRatings': sentinel.iso})
        expected = sentinel.iso

        assert actual == expected

    def test_raises_on_missing_tag(self):
        with pytest.raises(KeyError):
            module._parse_iso({})


@pytest.fixture
def mock_exif_parse(mocker):
    mocker.patch.object(module, '_parse_date_time_original')
    mocker.patch.object(module, '_parse_iso')
    mocker.patch.object(module, '_parse_exposure_time')


class TestGetExifTags:
    def test_reads_sidecar_jpeg_exif(self, mocker, mock_exif_parse):
        mock_read_exif_tags = mocker.patch.object(module, '_read_exif_tags')

        module.get_exif_tags('mock_image_path.dng')

        mock_read_exif_tags.assert_called_with('mock_image_path.jpeg')

    def test_parses_tags_correctly(self, mocker):
        mock_parsed_tags = {
            'DateTimeOriginal': '2018:09:28 20:29:59',
            'ExposureTime': (599985, 1000000),
            'ISOSpeedRatings': sentinel.iso,
        }

        mocker.patch.object(module, '_read_exif_tags').return_value = mock_parsed_tags

        actual = module.get_exif_tags('mock_image_path.dng')

        expected = module.ExifTags(
            capture_datetime=datetime(2018, 9, 28, 20, 29, 59),
            iso=sentinel.iso,
            exposure_time=599985/1000000
        )

        assert actual == expected
