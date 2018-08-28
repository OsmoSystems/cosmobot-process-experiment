# import the necessary packages
from pyzbar import pyzbar
import argparse
import cv2
from patch import detectPatches
from barcode import findBarcodeInImage

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

imageFilename = args["image"]
img = cv2.imread(imageFilename)

barcodeValue = findBarcodeInImage(img)

if(barcodeValue == None):
	print("No barcode Information found")
else:
	print("Barcode '{}' detected".format(barcodeValue))
	detectPatches(img, barcodeValue)
