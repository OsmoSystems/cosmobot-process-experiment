import numpy as np
from PIL import Image


# TODO: this name doesn't really make sense since I had to pass in the .png path
def as_png(rgb_image, image_path):
    # PIL Image expects a unsigned int 8-bit image
    rgb_image_as_uint_array = (rgb_image * (2 ** 8 - 1)).astype(np.uint8)
    img = Image.fromarray(rgb_image_as_uint_array, 'RGB')
    img.save(image_path)
