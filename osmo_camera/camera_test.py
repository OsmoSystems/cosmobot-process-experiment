'''Camera capture'''
from os import remove
from pathlib import Path
from camera import capture


def capture_test():
    '''Test that a file is created with capture'''
    filename = '../output/capture_test.jpeg'
    capture(filename)
    test_file = Path(filename)
    assert test_file.is_file()
    remove(filename)


def capture_test_with_additional_capture_params():
    '''Test that a file is created with capture'''
    filename = '../output/capture_test.jpeg'
    additional_capture_params = ' -ss 100 -ISO 100'
    capture(filename, additional_capture_params)
    test_file = Path(filename)
    assert test_file.is_file()
    remove(filename)
