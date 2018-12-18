import numpy as np


def _apply_flat_field_correction(dark_frame_corrected_rgb, dark_frame_rgb, flat_field_rgb):
    # TODO (https://app.asana.com/0/819671808102776/926723356906177): implement
    return dark_frame_corrected_rgb


def apply_flat_field_correction_to_rgb_images(rgbs_by_filepath):
    flat_field_corrected_rgbs_by_filepath = {
        image_path: _apply_flat_field_correction(
            image_rgb,
            dark_frame_rgb=np.zeros(image_rgb.shape),
            flat_field_rgb=np.ones(image_rgb.shape)
        )
        for image_path, image_rgb in rgbs_by_filepath.items()
    }

    return flat_field_corrected_rgbs_by_filepath
