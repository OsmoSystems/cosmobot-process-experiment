import io
import time
import picamera
import cv2
import numpy as np

# Create the in-memory stream
stream = io.BytesIO()

# create camera object
camera = picamera.PiCamera()
camera.start_preview()

def captureImage:
    camera.capture(stream, format='bgr')
	# Construct a numpy array from the stream
	data = np.fromstring(stream.getvalue(), dtype=np.uint8)

	# "Decode" the image from the array, preserving colour
	image = cv2.imdecode(data, 1)

	# OpenCV returns an array with data in BGR order. Reverse for RGB instead
	return image[:, :, ::-1]
