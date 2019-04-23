import pandas as pd

from osmo_camera.correction import dark_frame


def from_rgb_images(rgbs_by_filepath):
    ''' Generates a flat field `RGB image` from a set of `RGB images` based:
        https://docs.google.com/document/d/1i9VMA-XDHvCUdx-Bc7z1QxZqyYAOHiqckSKiXJOi0oU/edit

    Args:
        rgbs_by_filepath: A pandas Series of RGB images indexed by file path

    Returns:
        A flat field `RGB image` to be used for flat field correction
    '''

    # Subtract dark frame (using exposure of image) for all images
    dark_frame_adjusted_rgb_images = pd.Series({
        image_path: dark_frame.get_metadata_and_apply_dark_frame_correction(rgb_image, image_path)
        for image_path, rgb_image in rgbs_by_filepath.items()
    })

    # Average (image) of all dark frame adjusted images
    flat_field_mean_of_image_stack = dark_frame_adjusted_rgb_images.mean(axis=0)

    # Average RGB value of averaged image
    flat_field_mean = flat_field_mean_of_image_stack.mean(axis=(0, 1))

    return flat_field_mean / flat_field_mean_of_image_stack
