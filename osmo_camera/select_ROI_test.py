from osmo_camera import select_ROI as module


class TestPrettifyRoiDefinitions:
    def test_prettifies_empty_roi_definitions(self):
        assert module._prettify_roi_defintions({}) == "{\n\n}"

    def test_roi_definitions(self):
        ROI_definitions = {"DO patch": (100, 200, 300, 400)}
        expected = """{
    "DO patch": (100, 200, 300, 400),
}"""
        assert module._prettify_roi_defintions(ROI_definitions) == expected
