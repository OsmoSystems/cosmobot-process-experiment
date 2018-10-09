import os


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
