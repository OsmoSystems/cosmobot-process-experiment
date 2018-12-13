from datetime import datetime
from unittest.mock import sentinel

import pytest

from . import file_structure as module


@pytest.fixture
def mock_path_exists(mocker):
    return mocker.patch('os.path.exists')


@pytest.fixture
def mock_path_join(mocker):
    return mocker.patch('os.path.join')


@pytest.fixture
def mock_makedirs(mocker):
    return mocker.patch('os.makedirs')


@pytest.fixture
def mock_listdir(mocker):
    return mocker.patch('os.listdir')


class TestCreateOuputDirectory:
    def test_returns_output_directory_path(self, mock_path_exists, mock_path_join, mock_makedirs):
        mock_path_join.return_value = sentinel.expected_output_directory_path

        actual_output_directory_path = module.create_output_directory('/foo/bar', 'baz')

        assert actual_output_directory_path == sentinel.expected_output_directory_path

    def test_creates_output_directory_if_doesnt_exist(self, mock_path_exists, mock_path_join, mock_makedirs):
        mock_path_exists.return_value = False
        mock_path_join.return_value = sentinel.expected_output_directory_path

        module.create_output_directory('/foo/bar', 'baz')

        mock_makedirs.assert_called_with(sentinel.expected_output_directory_path)

    def test_doesnt_create_output_directory_if_exists(self, mock_path_exists, mock_path_join, mock_makedirs):
        mock_path_exists.return_value = True

        module.create_output_directory('/foo/bar', 'baz')

        mock_makedirs.assert_not_called()


def _mock_path_join(path, filename):
    return f'os-path-for-{path}/{filename}'


@pytest.mark.parametrize("name, directory_contents", [
    ('returns filepaths', ['1.jpeg', '2.jpeg']),
    ('sorts filepaths', ['2.jpeg', '1.jpeg']),
    ('filters to extension', ['1.jpeg', '2.jpeg', '1.dng', '2.png']),
])
def test_get_files_with_extension(mock_listdir, mock_path_join, name, directory_contents):
    mock_listdir.return_value = directory_contents
    mock_path_join.side_effect = _mock_path_join

    actual = module.get_files_with_extension('/foo/bar', '.jpeg')
    expected = ['os-path-for-/foo/bar/1.jpeg', 'os-path-for-/foo/bar/2.jpeg']

    assert actual == expected


class TestIsoDatetimeForFilename:
    def test_returns_iso_ish_string(self):
        actual = module.iso_datetime_for_filename(datetime(2018, 1, 2, 13, 14, 15))
        expected = '2018-01-02--13-14-15'

        assert actual == expected

    def test_result_length_matches_constant(self):
        actual = module.iso_datetime_for_filename(datetime(2018, 1, 2, 13, 14, 15))

        assert len(actual) == module.FILENAME_TIMESTAMP_LENGTH


class TestIsoDatetimeAndRestFromFilename:
    def test_returns_datetime(self):
        actual = module.datetime_from_filename('2018-01-02--13-14-15-something-something.jpeg')
        expected = datetime(2018, 1, 2, 13, 14, 15)

        assert actual == expected


class TestFilenameHasFormat:

    @pytest.mark.parametrize("filename, truthiness", [
        ('2018-01-02--13-14-15-something-something.jpeg', True),
        ('2018-01-02--13-aa-15-something-something.jpeg', False),
        ('2018-01-02--13-14-1-hi-hi.jpeg', False),
        ('prefix-2018-01-02--13-14-15something-something.jpeg', False),
    ])
    def test_filename_has_correct_datetime_format(self, filename, truthiness):
        assert module.filename_has_correct_datetime_format(filename) is truthiness


def test_append_suffix_to_filepath_before_extension():
    suffix = '_i_am_a_filepath_suffix'
    actual = module.append_suffix_to_filepath_before_extension('/dir/dir/image.jpeg', suffix)
    expected = '/dir/dir/image_i_am_a_filepath_suffix.jpeg'

    assert actual == expected
