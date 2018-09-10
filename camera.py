# import io
# from picamera import PiCamera
# from cv2 import imdecode
# import numpy as np
# from np import fromstring
from subprocess import call
# Create the in-memory stream
# stream = io.BytesIO()


# create camera object
# camera = PiCamera()
# camera.awb_mode = 'auto'
# camera.brightness = 50
# camera.exposure_mode = 'night'
# camera.resolution = (2592, 1944)
# sleep(2)
# raspistill --raw -hf -o ./raw_hf_flag.jpg
def captureImage(filename, in_format='jpeg', additional_capture_params=''):
    comm = 'raspistill --raw -o {} {}'.format(filename, additional_capture_params)
    print(comm)
    call([comm], shell=True)
    # camera.capture(filename, format=in_format)

# captureImage('./output/testyuv.raw', 'yuv')
