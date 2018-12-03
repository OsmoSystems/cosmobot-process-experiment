import pytest
import numpy as np
from osmo_camera import dng, rgb
from osmo_camera.correction import dark_frame, flat_field, intensity

from osmo_camera import blueness as module


@pytest.fixture
def mock_correction_effects(mocker):
    mocker.patch.object(dng.open, 'as_rgb').return_value = np.array([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
    ])

    mocker.patch.object(rgb.average, 'spatial_average_of_roi').return_value = None
    mocker.patch.object(dark_frame, 'apply_dark_frame_correction').return_value = None
    mocker.patch.object(flat_field, 'apply_flat_field_correction').return_value = None
    mocker.patch.object(intensity, 'apply_intensity_correction').return_value = None


def test_images_to_bluenesses(mock_correction_effects):
    dng_image_paths = ['/1.dng', '/2.dng']

    ROI_definitions = [
        [0, 0, 1, 1]
    ]

    ROI_for_intensity_correction = [1, 2, 3, 4]

    bluenesses = module.images_to_bluenesses(
        dng_image_paths,
        ROI_definitions,
        ROI_for_intensity_correction
    )

    assert bluenesses == {'/1.dng': 0.8, '/2.dng': 0.8}
