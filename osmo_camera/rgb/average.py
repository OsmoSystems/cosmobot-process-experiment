from . import crop


def spatial_average_of_roi(input_rgb, roi):
    image_crop = crop.crop_image(input_rgb, roi)
    return image_crop.mean()
