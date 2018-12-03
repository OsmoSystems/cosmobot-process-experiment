from . import crop

IMAGE_AXIS = (0, 1)


def spatial_average_of_roi(input_rgb, ROI):
    image_crop = crop.crop_image(input_rgb, ROI)
    return image_crop.mean(axis=IMAGE_AXIS)
