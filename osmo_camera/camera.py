'''Camera capture'''
from subprocess import call, check_output
import platform

# If distribution is debian than we assume that the python script is running on the pi
# and that a camera module is present.  If the distribution is not debian
# then we assume local development is occurring and to simulate a camera capture
# by copying an image file similar to how a camera capture occurs
# TODO: deprecated, better method?  platform.system() returns Linux on the pi
# whereas platform.dist()[0] returns debian
DIST = platform.dist()[0]
DEBIAN_DIST = 'debian'
LOCAL_CP_COMMAND = 'cp ./image_for_development.jpeg {}'


def capture(filename, additional_capture_params=''):
    '''Capture raw image JPEG+EXIF using command line'''
    comm = 'raspistill --raw -o {} {}'.format(filename, additional_capture_params)

    # if not on raspberry pi perform local copy command to simulate camera capture
    if DIST != DEBIAN_DIST:
        comm = LOCAL_CP_COMMAND.format(filename)

    print(f'Capturing image using raspistill: {comm}')

    call([comm], shell=True)

    comm_output = check_output(comm, shell=True).decode("utf-8")

    return comm_output
