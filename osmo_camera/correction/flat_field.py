import numpy as np
from osmo_camera import rgb

filepath_suffix = "_dark_flat_adj"


def _apply_flat_field_correction(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    # TODO (https://app.asana.com/0/819671808102776/926723356906177): implement
    return dark_frame_corrected_rgb


def apply_flat_field_correction_to_rgb_images(dark_frame_corrected_rgb_by_filepath, save_corrected_images=False):
    flat_field_corrected_rgb_by_filepath = {
        image_path: _apply_flat_field_correction(
            dark_frame_corrected_rgb,
            dark_frame_rgb=np.zeros(dark_frame_corrected_rgb.shape),
            flat_field_rgb=np.ones(dark_frame_corrected_rgb.shape)
        )
        for image_path, dark_frame_corrected_rgb in dark_frame_corrected_rgb_by_filepath.items()
    }

    if save_corrected_images:
        rgb.save.save_rgb_images_by_filepath_with_suffix(flat_field_corrected_rgb_by_filepath, filepath_suffix)

    return flat_field_corrected_rgb_by_filepath
