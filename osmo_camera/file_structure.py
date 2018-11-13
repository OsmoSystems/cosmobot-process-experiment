import datetime
import os


_FILENAME_DATETIME_FORMAT = '%Y-%m-%d--%H-%M-%S'
FILENAME_TIMESTAMP_LENGTH = len('2018-01-01--12-01-01')


def get_files_with_extension(directory, extension):
    ''' Get all file paths in the given directory with the given extension, sorted alphanumerically

    Args:
        directory: The full path to the directory of files
        extension: The full extension (including '.') of files to filter to, e.g. '.jpeg'

    Returns:
        A sorted list of full file paths
    '''
    file_paths = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if os.path.splitext(filename)[1] == extension  # splitext() splits into a tuple of (root, extension)
    ]

    return sorted(file_paths)


def create_output_directory(base_directory, new_directory_name):
    '''Create a new directory with a base path if it does not exist'''
    new_directory_path = os.path.join(base_directory, new_directory_name)
    return create_directory(new_directory_path)


def create_directory(new_directory_path):
    '''Create a new directory if it does not exist'''
    if not os.path.exists(new_directory_path):
        print(f'creating folder: {new_directory_path}')
        os.makedirs(new_directory_path)

    return new_directory_path


def iso_datetime_for_filename(datetime):
    ''' Returns datetime as a ISO-ish format string that can be used in filenames (which can't inclue ":")
    iso_datetime_for_filename(datetime.datetime(2018, 1, 1, 12, 1, 1)) -> '2018-01-01--12-01-01'
    '''
    return datetime.strftime(_FILENAME_DATETIME_FORMAT)


def datetime_from_filename(filename: str) -> datetime.datetime:
    ''' Recover a datetime that has been encoded into a filename, also returning the remainder of the filename

    Arguments:
        filename: filename to process. Should start with an ISO-ish datetime
            as produced by iso_datetime_for_filename()
    Returns:
        a datetime.datetime object matching the one that was encoded in the filename
    '''

    return datetime.datetime.strptime(filename[:FILENAME_TIMESTAMP_LENGTH], _FILENAME_DATETIME_FORMAT)
