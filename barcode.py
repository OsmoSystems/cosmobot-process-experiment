# import the necessary packages
from pyzbar import pyzbar
import cv2

def findBarcodeInImage(img):
    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(img)
    barcodeValue = None
    barcodeType = None

    # loop over the detected barcodes
    for barcode in barcodes:
        barcodeValue = barcode.data.decode("utf-8")
        barcodeType = barcode.type

    return barcodeValue
