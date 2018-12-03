import pytest
import numpy as np
from osmo_camera import dng, rgb
from osmo_camera.correction import dark_frame, flat_field, intensity

from osmo_camera import blueness as module


@pytest.fixture
def mock_correction_effects(mocker):
    mocker.patch.object(dng.open, 'as_rgb')
    mocker.patch.object(rgb.average, 'spatial_average_of_roi').return_value = [0]
    mocker.patch.object(dark_frame, 'dark_frame_correction').return_value = [0]
    mocker.patch.object(flat_field, 'flat_field_correction').return_value = [0]
    mocker.patch.object(intensity, 'intensity_correction').return_value = np.array([
        [11, 12, 13, 14],
        [15, 16, 17, 18],
        [19, 20, 21, 22]
    ])


def test_images_to_bluenesses(mock_correction_effects):
    dng_image_paths = ['/1.dng', '/2.dng']

    roi_for_blueness = [0, 0, 1, 1]

    roi_for_intensity_correction = [1, 2, 3, 4]

    bluenesses = module.images_to_bluenesses(
        dng_image_paths,
        roi_for_blueness,
        roi_for_intensity_correction
    )

    print(bluenesses)

    assert True
