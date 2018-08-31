'''Process Image'''

import argparse
from datetime import datetime
from cv2 import imread # pylint: disable=E0611
from patch import extract_patches_from_image
from barcode import extract_barcode_from_greyscale_image

# construct the argument parser and parse the arguments
AP = argparse.ArgumentParser()
AP.add_argument("-i", "--image", required=True, help="path to input image")
ARGS = vars(AP.parse_args())

def process_image(input_filename):
    '''Process an image of a cartridge'''
    input_image = imread(input_filename)
    input_image_greyscale = imread(input_filename, 0)
    barcode_value = extract_barcode_from_greyscale_image(input_image_greyscale)

    output_file_prefix = datetime.now().strftime('%Y%m%d%H%M%S')

    if barcode_value is None:
        print("No barcode Information found")
    else:
        print("Barcode '{}' detected".format(barcode_value))
        extract_patches_from_image(input_image, barcode_value, output_file_prefix)

process_image(ARGS["image"])
