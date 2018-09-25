'''Perform camera capture experiment'''

from shutil import disk_usage

def test_can_store_experiment(duration, interval):
    '''dd'''
    average_image_size_in_bytes = 1024 * 1024 * 1024
    total = disk_usage('/')
    print(total)
    return total / 1024 / 1024 / 1024

print(can_store_experiment(1,1))
