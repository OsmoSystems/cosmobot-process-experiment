import os

# TODO: better name for this module now that it includes great_output_directory


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
        os.mkdir(new_directory_path)

    return new_directory_path
