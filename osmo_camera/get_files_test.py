import pytest

from . import get_files as module


@pytest.fixture
def mock_exists(mocker):
    return mocker.patch('os.path.exists')


@pytest.fixture
def mock_mkdir(mocker):
    return mocker.patch('os.mkdir')


@pytest.fixture
def mock_listdir(mocker):
    return mocker.patch('os.listdir')


class TestCreateOuputDirectory:
    def test_returns_output_directory_path(self, mock_exists, mock_mkdir):
        output_directory_path = module.create_output_directory('/foo/bar', 'baz')

        assert output_directory_path == '/foo/bar/baz'

    def test_creates_output_directory_if_doesnt_exist(self, mock_exists, mock_mkdir):
        mock_exists.return_value = False

        module.create_output_directory('/foo/bar', 'baz')

        mock_mkdir.assert_called_with('/foo/bar/baz')

    def test_doesnt_create_output_directory_if_exists(self, mock_exists, mock_mkdir):
        mock_exists.return_value = True

        module.create_output_directory('/foo/bar', 'baz')

        mock_mkdir.assert_not_called()


class TestGetFilesWithExtension:
    def test_returns_full_paths(self, mock_listdir):
        mock_listdir.return_value = ['1.jpeg', '2.jpeg']

        actual = module.get_files_with_extension('/foo/bar', '.jpeg')
        expected = ['/foo/bar/1.jpeg', '/foo/bar/2.jpeg']

        assert actual == expected

    def test_filters_to_extension(self, mock_listdir):
        mock_listdir.return_value = ['1.jpeg', '2.jpeg', '1.dng', '2.dng']

        actual = module.get_files_with_extension('/foo/bar', '.jpeg')
        expected = ['/foo/bar/1.jpeg', '/foo/bar/2.jpeg']

        assert actual == expected

    def test_returns_sorted_list(self, mock_listdir):
        mock_listdir.return_value = ['1.jpeg', '3.jpeg', '2.jpeg']

        actual = module.get_files_with_extension('/foo/bar', '.jpeg')
        expected = ['/foo/bar/1.jpeg', '/foo/bar/2.jpeg', '/foo/bar/3.jpeg']

        assert actual == expected
