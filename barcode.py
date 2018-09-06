'''Utility module for barcode extraction from an image'''
from pyzbar import pyzbar


def extract_barcode_from_greyscale_image(input_image_greyscale):
    '''find the barcodes in the image and decode each of the barcodes'''
    barcodes = pyzbar.decode(input_image_greyscale)
    barcode_value = None

    for barcode in barcodes:
        barcode_value = barcode.data.decode("utf-8")

    return barcode_value
