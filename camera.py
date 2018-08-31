import io
from picamera import PiCamera
from cv2 import imdecode
import numpy as np
from np import fromstring
from time import sleep

# Create the in-memory stream
stream = io.BytesIO()

# create camera object
camera = PiCamera()
camera.start_preview()
sleep(2)

def captureImage():
    capture(stream, format='bgr')
	# Construct a numpy array from the stream
    data = fromstring(stream.getvalue(), dtype=np.uint8)

	# "Decode" the image from the array, preserving colour
    image = imdecode(data, 1)

	# OpenCV returns an array with data in BGR order. Reverse for RGB instead
    return image[:, :, ::-1]
