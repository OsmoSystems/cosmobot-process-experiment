import os


def get_files_with_extension(directory, extension):
    file_paths = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if os.path.splitext(filename)[1] == extension  # splitext() splits into a tuple of (root, extension)
    ]

    return sorted(file_paths)
