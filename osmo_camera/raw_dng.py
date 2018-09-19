'''
# Rough overview of image processing algorithm
# 0) raspistill + raspidng used for camera raw image capture and dng extraction
# 1) rawpy used for extracting bayer raw image and color information from a DNG file (Adobe digital negative)
# 2) extract r, g, b channels using raw_image data and bayer color information
# 3) transform and return r,g,b channels

# Dependencies:
# rawpy (py wrapper for LibRaw, which is a wrapper for dcraw) https://github.com/letmaik/rawpy, https://www.libraw.org/,
# http://cybercom.net/~dcoffin/dcraw/
# rawpy used to extract the raw image and color information
# imageio used for writing images
# numpy for array manipulation

# A note on Bayer image data https://picamera.readthedocs.io/en/release-1.13/recipes2.html#raw-bayer-data-captures
# Bayer image data is represented as a two dimension array (pixel y, pixel x) of color counts per channel.
# These values represent unprocessed data in that they are the most unadulterated signal coming from the CMOS sensor
# that we have access to.  The data in each index of the two dimensional array represents either a R, G, B value at
# a given pixel.  Whether the value is representing an R, G, B value is explained below.
Example Bayer image data shape:
[[ 896  960  832 ... 1088 1088 1664]
 [1280  704  640 ... 1216 1152 1536]
 [1472 1088  768 ... 1408 1408 1024]
 ...
 [ 576  576  704 ...  768 1216 1088]
 [1024  576  640 ... 1088  896 1088]
 [ 640  640  896 ...  896 1472 1472]]

# Bayer color data:
# Bayer color data is represented as a two dimension array (pixel y, pixel x) that essentially represents what color
# channel a
# value represents in the bayer image data.  The shape of both the bayer image data and the bayer color data are
# identical,
# as a value at a location in the raw color array is used to inform which color channel that same location in the raw
# image array represents.
# Based on testing outputs of images, the following values represent each color channel # (TODO: better way to verify?)
# 0 represents red
# 1 represents green
# 2 represents blue
# 3 represents a second green sensor
Example raw color data shape:           Converted to RGBG
[[2 3 2 ... 3 2 3]                      [[B G B ... G B G]
 [1 0 1 ... 0 1 0]                       [G R G ... G R G]...
 [2 3 2 ... 3 2 3]
 ...
 [1 0 1 ... 0 1 0]
 [2 3 2 ... 3 2 3]
 [1 0 1 ... 0 1 0]]

# Half resolution of R, G, B channel data
# There are less R, G, B "pixels" than a resolution of a processed image reports.
# There are half the number of R & B filtered photodiodes then that of the eventual processed image.
# There are twice the number of G filtered photodiodes than R & B due to how the CMOS sensor is constructed towards a
# human eye's
# representation of an image - https://www.cambridgeincolour.com/tutorials/camera-sensors.htm
# "Each primary color does not receive an equal fraction of the total area because the human eye is more sensitive to
# green light
# than both red and blue light.  Redundancy with green pixels produces an image which appears less noisy and has finer
# detail than
# could be accomplished if each color were treated equally."

# Verfication of algorithm - there are two ways in which values were verfified:
1) Compare rgb values of a pixel with the algorithm to the rgb picker in RawDigger program.  Repeat for multiple pixels
in image.
2) Reconstruct image and verify through an eye-smell test that the image looks correct
3) TODO: better validation of algorihm?
'''

import numpy as np
import rawpy
import exifread
import os
import datetime


def get_create_date(filename):
    '''filename is in an EXIF key formatted like
    'EXIF DateTimeOriginal': (0x0132) ASCII=2018:09:10 20:01:19 @ 59140
    the right side is an EXIF key value; getting ex_key.values
    gives you a nice ISO8601-ish string
    '''

    filename_prefix, file_extension = os.path.splitext(filename)
    jpeg_filename = f'{filename_prefix}.jpeg'
    desired_tag = 'EXIF DateTimeOriginal'

    with open(jpeg_filename, 'rb') as fh:
        tags = exifread.process_file(fh)

    date_taken = tags[desired_tag]
    return datetime.datetime.strptime(date_taken.values, '%Y:%m:%d %H:%M:%S')


def color_channels_from_raw_dng(filename, fix_flipped_colors=False):
    ''' Get red, green amd blue color channels

    Arguments:
        filename: name of the file (usually a .dng) to process
        fix_flipped_colors: boolean, whether to fix an apparent issue with raspberry pi raspiraw DNG files where the
            bayer filter map is flipped left-to-right.

    Returns:
        dict of {red: red grayscale image, green: green grayscale image, blue: blue grayscale image} where each
            grayscale image is a numpy 2D array of integers from the raw data.
    '''
    # copy bayer raw_image data and raw_color information (bayer raw image data and bayer raw color patterns)
    with rawpy.imread(filename) as raw:
        raw_image = raw.raw_image.copy()
        original_colors = raw.raw_colors.copy()
        raw_colors = np.fliplr(original_colors) if fix_flipped_colors else original_colors

    image_width = raw_colors.shape[1]
    image_height = raw_colors.shape[0]

    # the width and height of each color channel is half that of the full image
    half_image_width = int(image_width / 2)
    half_image_height = int(image_height / 2)

    # flatten image and color data to linear array to extract color data by channel
    raw_img = raw_image.flatten()
    raw_clr = raw_colors.flatten()

    # TODO: research if second green channel should be combined somehow to improve our green count

    # retrieve indices of specific color from bayer raw_color
    red_indices = np.argwhere(raw_clr == 0)
    green_indices = np.argwhere(raw_clr == 1)
    blue_indices = np.argwhere(raw_clr == 2)

    # filter array with indices representing a specific color sensor
    # and reshape linear color array to 2 dimensional array that represents a value in a color channel at a pixel
    filtered_image_red_channel_yx = raw_img[red_indices].reshape(half_image_height, half_image_width)
    filtered_image_green_channel_yx = raw_img[green_indices].reshape(half_image_height, half_image_width)
    filtered_image_blue_channel_yx = raw_img[blue_indices].reshape(half_image_height, half_image_width)

    return dict(
        red=filtered_image_red_channel_yx,
        green=filtered_image_green_channel_yx,
        blue=filtered_image_blue_channel_yx
    )


def compose_rgb_channels_to_Y_X_RGB(red, green, blue):
    composed_rgb = np.dstack((red, green, blue))
    return composed_rgb


def compose_rgb_channels_to_opencv_format(red, green, blue):
    return np.dstack((blue, green, red)) / 2 ** 16


def open_image(filename):
    return compose_rgb_channels_to_opencv_format(
        **color_channels_from_raw_dng(
            filename,
            fix_flipped_colors=True
        )
    )


# tests :)
# color_channels = color_channels_from_raw_dng('./input/raw_hf_flag.dng')
# composed_rgb = compose_rgb_channels_to_Y_X_RGB(**color_channels)
# imageio.imwrite('./output.tiff', composed_rgb)
