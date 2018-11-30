from osmo_camera import dng, rgb, correction

import correction.dark_frame as dark_frame
import correction.dark_frame_correction, flat_field_correction
from correction import intensity_correction


def _calculate_blueness(input_rgb, roi):
    # TODO: Implement blueness calculation
    # image_crop = crop.crop_image(input_rgb, roi)
    return 0.8


def images_to_bluenesses(
    dng_image_paths,
    target_roi,
    intensity_correction_roi
):
    ''' Process all DNGs through dark frame, flat field, and intensity correction (Stubbed)

    Args:
        dng_image_paths: list of file paths for dngs to be processed into a "blue" value
        roi_for_blueness: the roi that is to be used for calculating the "blue" value
        roi_for_intensity_correction: region to average

    Returns:
        A dictionary of blue values that is keyed by dng file path
    '''

    dark_frame_adjusted_rgb = dict()
    flat_field_adjusted_rgb = dict()
    intensity_corrected_rgb = dict()
    blueness = dict()

    for image_path in dng_image_paths:
        dark_frame_adjusted_rgb[image_path] = dark_frame_correction(dng.open(image_path))

    for image_path in dng_image_paths:
        flat_field_adjusted_rgb[image_path] = flat_field_correction(dark_frame_adjusted_rgb[image_path])

    for image_path in dng_image_paths:
        intensity_correction_roi_spatial_average = rgb.average.spatial_average_of_roi(
            flat_field_adjusted_rgb,
            intensity_correction_roi
        )
        flat_field_adjusted_rgb[image_path] = intensity_correction(
            intensity_corrected_rgb,
            intensity_correction_roi_spatial_average
        )

    for image_path, corrected_rgb in intensity_corrected_rgb.items():
        blueness[image_path] = _calculate_blueness(corrected_rgb, target_roi)

    return blueness
