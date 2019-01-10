import pandas as pd


def get_intensity_correction_diagnostics(before, after, image_path):
    # TODO (https://app.asana.com/0/862697519982053/933995774091561): implement
    ''' Produce diagnostics and raise warnings based on an intensity-corrected image

    Documentation of individual diagnostics and warnings is in README.md in the project root.

    Args:
        before: RGB image before intensity correction
        after: RGB image after intensity correction
        image_path: path to original raw image. Used to look up EXIF data
    Returns:
        pandas Series of diagnostics and "red flag" warnings.
    Warns:
        uses the Warnings API and CorrectionWarning if any red flags are present.
    '''
    return pd.Series({})


def _apply_intensity_correction(input_rgb, ROI_definition_for_intensity_correction):
    # TODO (https://app.asana.com/0/862697519982053/933995774091561): implement
    return input_rgb


def apply_intensity_correction_to_rgb_images(rgbs_by_filepath, ROI_definition_for_intensity_correction):
    return rgbs_by_filepath.apply(
        _apply_intensity_correction,
        ROI_definition_for_intensity_correction=ROI_definition_for_intensity_correction
    )
