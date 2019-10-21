from osmo_camera import select_ROI as module

# Declaring this up here so that indentation doesn't make it too ugly xD
NICELY_FORMATTED_ROI = """{
    "DO patch": (100, 200, 300, 400),
}"""


class TestPrettifyRoiDefinitions:
    def test_prettifies_empty_roi_definitions(self):
        assert module._prettify_roi_defintions({}) == "{\n}"

    def test_roi_definitions(self):
        ROI_definitions = {"DO patch": (100, 200, 300, 400)}
        expected = NICELY_FORMATTED_ROI
        assert module._prettify_roi_defintions(ROI_definitions) == expected
