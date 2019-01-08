import pandas as pd


def get_intensity_correction_diagnostics(before, after, image_path):
    # TODO (https://app.asana.com/0/862697519982053/933995774091561): implement
    return pd.Series({})


def _apply_intensity_correction(input_rgb, ROI_definition_for_intensity_correction):
    # TODO (https://app.asana.com/0/862697519982053/933995774091561): implement
    return input_rgb


def apply_intensity_correction_to_rgb_images(rgbs_by_filepath, ROI_definition_for_intensity_correction):
    return rgbs_by_filepath.apply(
        _apply_intensity_correction,
        ROI_definition_for_intensity_correction=ROI_definition_for_intensity_correction
    )
