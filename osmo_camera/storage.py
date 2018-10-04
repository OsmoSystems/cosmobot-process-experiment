from shutil import disk_usage


# Experimental evidence shows the raw image size on the Sony IMX Camera module
# to max out at 1600000 bytes in size
IMAGE_SIZE_IN_BYTES = 1600000


def _free_space_for_image_count(image_count):
    '''Check if there is enough space with the storage device
     Args:
        image_count: how many images will be stored
     Returns:
        Boolean: True/False - is there space to store the experiment
    '''
    _, _, free = disk_usage('/')
    return free > IMAGE_SIZE_IN_BYTES * image_count


def how_many_images_with_free_space():
    '''Estimate how many images can be stored on the storage device
     Args:
        None
     Returns:
        Boolean: How many images can be stored
    '''
    _, _, free = disk_usage('/')
    return free / IMAGE_SIZE_IN_BYTES


def free_space_for_one_image():
    '''Is there enough space for one image
     Args:
        None
     Returns:
        Boolean: True/False - is there space to store one image
    '''
    return _free_space_for_image_count(1)
