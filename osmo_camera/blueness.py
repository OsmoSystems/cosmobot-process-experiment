from osmo_camera import dng, rgb
from osmo_camera.correction import dark_frame, flat_field, intensity


def _calculate_blueness(input_rgb, roi):
    # TODO: Implement blueness calculation
    # image_crop = crop.crop_image(input_rgb, roi)
    return 0.8


def images_to_bluenesses(
    dng_image_paths,
    roi_for_blueness,
    roi_for_intensity_correction,
):
    ''' Process all DNGs through dark frame, flat field, and intensity correction (Stubbed)

    Args:
        dng_image_paths: list of file paths for dngs to be processed into a "blue" value
        roi_for_blueness: the roi that is to be used for calculating the "blue" value
        roi_for_intensity_correction: region to average

    Returns:
        A dictionary of blue values that is keyed by dng file path
    '''

    dark_frame_corrected_rgb_by_filepath = dict()
    flat_field_corrected_rgb_by_filepath = dict()
    intensity_corrected_rgb_by_filepath = dict()
    blueness_by_filepath = dict()

    # open all images and perform dark frame correction
    for image_path in dng_image_paths:
        image_rgb = dng.open.as_rgb(image_path)
        dark_frame_rgb = image_rgb  # TODO: add retrieval later
        dark_frame_corrected_rgb_by_filepath[image_path] = dark_frame.dark_frame_correction(image_rgb, dark_frame_rgb)

    # perform flat field correction on all images
    for image_path in dng_image_paths:
        dark_frame_corrected_rgb = dark_frame_corrected_rgb_by_filepath[image_path]
        dark_frame_rgb = dark_frame_corrected_rgb
        flat_field_rgb = dark_frame_corrected_rgb  # TODO: add retrieval later

        flat_field_corrected_rgb_by_filepath[image_path] = flat_field.flat_field_correction(
            dark_frame_corrected_rgb,
            dark_frame_rgb,
            flat_field_rgb
        )

    # perform intensity correction on all images
    for image_path in dng_image_paths:
        intensity_correction_roi_spatial_average = rgb.average.spatial_average_of_roi(
            flat_field_corrected_rgb_by_filepath[image_path],
            roi_for_intensity_correction
        )

        intensity_corrected_rgb_by_filepath[image_path] = intensity.intensity_correction(
            flat_field_corrected_rgb_by_filepath[image_path],
            intensity_correction_roi_spatial_average
        )

    # calculate those pesky blue values!
    for image_path, corrected_rgb in intensity_corrected_rgb_by_filepath.items():
        blueness_by_filepath[image_path] = _calculate_blueness(corrected_rgb, roi_for_blueness)

    return blueness_by_filepath
