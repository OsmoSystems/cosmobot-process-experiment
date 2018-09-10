'''Camera capture'''
from subprocess import call

import platform

# If distribution is debian than we assume that the python script is running on the pi
# and that a camera module is present.  If the distribution is not debian
# then we assume local development is occurring and to simulate a camera capture
# by copying an image file similar to how a camera capture occurs
DIST = platform.dist()[0]
DEBIAN_DIST = 'debian'
LOCAL_CP_COMMAND = 'cp ./image_for_development.jpeg {}'


def capture_image(filename):
    '''Capture raw image JPEG+EXIF using command line'''
    comm = 'raspistill --raw -o {}'.format(filename)

    # if not on raspberry pi perform local copy command to simulate camera capture
    if DIST != DEBIAN_DIST:
        comm = LOCAL_CP_COMMAND.format(filename)

    print("Capturing image using raspistill: {}".format(comm))

    call([comm], shell=True)
