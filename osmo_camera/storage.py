import math
from shutil import disk_usage
# Experimental evidence shows the raw image size on the Sony IMX Camera module
# to max out at 1600000 bytes in size
IMAGE_SIZE_IN_BYTES = 1600000


def estimate_image_count(duration, interval, seconds_for_image_capture=5):
    '''Estimate how many images will be captured with interval and duration
     Args:
        duration: seconds the experiment will run for
        interval: interval in seconds between image capture
        seconds_for_image_capture: estimated time it takes for one image to be taken
     Returns:
        float: How many images can be stored
    '''
    return int(math.floor(duration / seconds_for_image_capture))


def how_many_images_with_free_space():
    '''Estimate how many images can be stored on the storage device
     Args:
        None
     Returns:
        Boolean: How many images can be stored
    '''
    _, _, free = disk_usage('/')
    return free / IMAGE_SIZE_IN_BYTES


def free_space_for_image_count(image_count):
    '''Check if there is enough space with the storage device
     Args:
        image_count: how many images will be stored
     Returns:
        Boolean: True/False - is there space to store the experiment
    '''
    _, _, free = disk_usage('/')
    return free > IMAGE_SIZE_IN_BYTES * image_count


def free_space_for_one_image():
    '''Is there enough space for one image
     Args:
        None
     Returns:
        Boolean: True/False - is there space to store one image
    '''
    return free_space_for_image_count(1)


def free_space_for_experiment(duration, interval, seconds_for_image_capture=5):
    '''Is there enough space for the entire experiment
     Args:
        duration: seconds the experiment will run for
        interval: interval in seconds between image capture
        seconds_for_image_capture: estimated time it takes for one image to be taken
     Returns:
        Boolean: True/False - is there space to store experiment
    '''
    # TODO: assumes capture takes no time
    image_count = duration / interval * IMAGE_SIZE_IN_BYTES
    return free_space_for_image_count(image_count)
