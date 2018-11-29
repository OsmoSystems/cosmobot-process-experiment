from dng import open
from rgb import spatial_average_of_roi #, crop
from correction import dark_frame_correction, flat_field_correction
from correction import intensity_correction


def _calculate_blueness(input_rgb, roi):
    # TODO: Implement blueness calculation
    # image_crop = crop.crop_image(input_rgb, roi)
    return 0.8


def images_to_bluenesses(
    # TODO: should dng_image_paths be a a tuple that contains an roi and path?
    # will images have separate ROIs?
    dng_image_paths,
    roi_for_blueness,
    rois_for_intensity_correction=None,
):
    '''WIP
    In general the pipeline follows Option #2, all images processed through a step before
    progressing to the next oneself.
    '''

    # TODO: Check for memory allocation based on number of DNG's and provide warning message?

    # TODO: check if correct data structure?
    processed_dngs = dict()

    output_template = dict(
        dark_frame_adjusted_rgb=None,
        flat_field_adjusted_rgb=None,
        intensity_corrected_rgb=None,
        blueness=None
    )

    for image_path in dng_image_paths:
        processed_dngs['dng_image_path'] = output_template.copy()

    for image_path in processed_dngs:
        try:
            processed_dngs[image_path]['dark_frame_adjusted_rgb'].append(dark_frame_correction(open(image_path)))
        except ValueError:
            processed_dngs[image_path]['dark_frame_adjusted_rgb'].append(None)

    for image_path in processed_dngs:
        dark_frame_adjusted_rgb = processed_dngs[image_path]['dark_frame_adjusted_rgb']

        try:
            if dark_frame_adjusted_rgb is None:
                processed_dngs[image_path]['flat_field_adjusted_rgb'] = None
                continue

            processed_dngs[image_path]['flat_field_adjusted_rgb'] = flat_field_correction(dark_frame_adjusted_rgb)

        except ValueError:
            processed_dngs['flat_field_adjusted_rgb'].append(None)


    for image_path in processed_dngs:
        flat_field_adjusted_rgb = processed_dngs[image_path]['flat_field_adjusted_rgb']

        try:
            if flat_field_adjusted_rgb is None:
                processed_dngs[image_path]['intensity_corrected_rgb'] = None
                continue

            # Will this ever happen?
            if rois_for_intensity_correction is None:
                processed_dngs[image_path]['intensity_corrected_rgb'] = flat_field_adjusted_rgb
                continue

            intensity_corrected_rgb = flat_field_adjusted_rgb


            for roi in rois_for_intensity_correction:
                roi_spatial_average = spatial_average_of_roi(flat_field_adjusted_rgb, roi)
                # TODO: we should probably store and have the ability to output every intensity
                # correction iteration, right now we only store the final intensity corrected
                # image
                intensity_corrected_rgb = intensity_correction(
                    intensity_corrected_rgb,
                    roi_spatial_average
                )

            processed_dngs[image_path]['intensity_corrected_rgb'] = intensity_corrected_rgb

        except ValueError:
            processed_dngs[image_path]['intensity_corrected_rgb'] = None

    for image_path in processed_dngs:
        adjusted_image = processed_dngs[image_path]['intensity_corrected_rgb']
        try:
            if adjusted_image is None:
                processed_dngs[image_path]['blueness'] = None
                continue

            processed_dngs[image_path]['blueness'] = _calculate_blueness(adjusted_image, roi_for_blueness)

        except ValueError:
            processed_dngs[image_path]['blueness'] = None

    return processed_dngs
