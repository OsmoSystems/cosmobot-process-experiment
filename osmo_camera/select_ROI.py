from ast import literal_eval
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from IPython.display import display
from ipywidgets import widgets

from osmo_camera import rgb
from osmo_camera.rgb.annotate import draw_ROIs_on_image


def _prettify_roi_defintions(ROI_definitions):
    """ Convert ROI definitions to a nice concise format for printing.
    Output is something like:
    {
        "DO patch": (1, 1, 100, 100),
    }
    """
    pretty_printed_keys_and_values = [
        f'\n    "{key}": {value},' for key, value in ROI_definitions.items()
    ]
    return "{" + "".join(pretty_printed_keys_and_values) + "\n}"


class ROISelectionInterface:
    """ UI for selecting ROIs on an image.
    To add an ROI:
    1. Click and drag with the mouse on the image
    2. Give the ROI a name in the ROIs text box
    3. Click outside of the text box to see updated labels on the image.
    You can also make arbitrary edits to ROIs using the text box.

    Args:
        image: RGB image
    Returns:
        ROISelectionInterface instance

    Attributes:
        ROI_definitions: dict of named regions of interest that have been selected
        get_image_with_rois(): returns a version of the input image with labeled ROI overlays

    Note: Using this forces matplotlib to use the notebook backend globally.
    You can revert this with `matplotlib.use(your preferred backend)`
    """

    def __init__(self, image):
        self.original_image = image
        self.ROI_definitions: Dict[str, Tuple[int, int, int, int]] = {}

        self._initialize_widgets()

        # Flag used to prevent triggering events from an update (-> unwanted recursion)
        self._is_updating = False
        self._update_output()

    def _initialize_widgets(self):
        """ Set up image display, ROI saving UI and ROI view/edit text box.
        Creates instance attributes that we'll reference elsewhere
        """
        matplotlib.use("nbAgg")  # Engage interactive mode. this is a global setting
        self.figure, self.axes = plt.subplots(
            figsize=(10, 8), num="ROI selection"  # num = Figure name in the UI.
        )
        self.figure.tight_layout(pad=0)

        self.roi_rectangle_selector = RectangleSelector(
            self.axes,
            self._handle_rectangle_change,
            drawtype="box",
            useblit=True,
            interactive=True,
            rectprops=dict(edgecolor="#00ff00", fill=False),
        )

        self.roi_text_box = widgets.Textarea(
            description="ROIs:",
            disabled=False,
            # Since input often goes through invalid states while the user is typing,
            # Only trigger the callback when user submits or changes focus away
            continuous_update=False,
        )
        self.roi_text_box.layout.height = "150px"
        self.roi_text_box.layout.width = "100%"
        self.roi_text_box.observe(self._handle_roi_text_box_change, names="value")
        display(self.roi_text_box)

    def _update_output(self):
        self._is_updating = True

        self.roi_text_box.value = _prettify_roi_defintions(self.ROI_definitions)

        # Show the old image before cleaning up the old one to prevent a period with no image
        self.axes.imshow(self.get_image_with_rois())
        # Clear old image, if any. If we let them pile up, the UI gets laggy
        for image in self.axes.get_images()[:-1]:
            image.remove()

        self._is_updating = False

    def get_image_with_rois(self):
        return draw_ROIs_on_image(self.original_image, self.ROI_definitions)

    def _get_current_roi(self):
        xmin, xmax, ymin, ymax = self.roi_rectangle_selector.extents
        # Round for cleanliness
        return (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))

    def _handle_roi_text_box_change(self, event):
        if not self._is_updating:
            self.ROI_definitions = literal_eval(self.roi_text_box.value)
            self._update_output()

    def _handle_rectangle_change(self, _, __):
        if not self._is_updating:
            self.current_ROI = self._get_current_roi()
            # Add placeholder ROI definition named with an empty string
            self.ROI_definitions[""] = self.current_ROI
            self._update_output()


def get_ROIs_for_image(rgb_image, ROI_definitions):
    ROIs = {
        ROI_name: rgb.crop.crop_image(rgb_image, ROI_definition)
        for ROI_name, ROI_definition in ROI_definitions.items()
    }

    return ROIs
