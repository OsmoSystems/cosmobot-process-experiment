from copy import deepcopy

import cv2

from osmo_camera import rgb


def choose_regions(rgb_image):
    ''' Funky interaction to select regions within an image.

    Arguments:
        rbg_image: 3D numpy.ndarray as an RGB image
    Returns:
        numpy 2d array, essentially an iterable containing iterables of (start_col, start_row, cols, rows)
        corresponding to the regions that you selected.
    '''

    print('''\n Working around some OpenCV bugs. Here's how to select ROIs:

    1. Go to the window that has popped up (likely behind other windows)
    2. Click + drag to select a region
    3. PRESS ENTER ONCE. (Pressing enter multiple times will save the same region again)
    4. Repeat steps 2-3 to select multiple regions
    5. Press ESC to complete the process. The window might not actually close :(
    ''')

    window_name = 'ROIs selection'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)  # WINDOW_GUI_EXPANDED seems to allow you to resize the window

    # Resize the window to a manageable default.
    window_size = 600  # in pixels
    cv2.resizeWindow(window_name, window_size, window_size)

    # Allows user to define ROIs (by selecting and pressing ENTER), until ESC is pressed to end selection process
    bgr_image = rgb.convert.to_bgr(rgb_image)  # OpenCV expects bgr format
    regions = cv2.selectROIs(window_name, bgr_image, showCrosshair=False)

    # OpenCV doesn't seem to like to actually close windows. Various attempts to force it have been unsuccessful.
    cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Wait for 1ms then (attempt to) close window. Supposedly necessary to "flush" the destroy events

    return regions


def prompt_for_ROI_selection(rgb_image):
    # Make image brighter to enable selecting ROIs even on very dark images
    brighter_rgb_image = rgb_image * 3
    ROIs = choose_regions(brighter_rgb_image)
    return ROIs


def input_ROI_names(ROIs):
    print('\nName your ROIs in the same order you selected them. Names must be unique.')
    ROI_definitions = {
        input(f'Unique name for ROI #{index + 1}: '): list(ROI)  # Convert np array to list to make print readable
        for index, ROI in enumerate(ROIs)
    }

    return ROI_definitions


def draw_ROIs_on_image(rgb_image, ROI_definitions={}):
    rgb_image_with_ROI_definitions = deepcopy(rgb_image)

    for ROI_name, ROI_definition in ROI_definitions.items():
        [start_col, start_row, cols, rows] = ROI_definition

        top_left_corner = (start_col, start_row)
        bottom_right_corner = (start_col + cols, start_row + rows)
        green_color = (0, 1, 0)

        cv2.rectangle(
            rgb_image_with_ROI_definitions,
            top_left_corner,
            bottom_right_corner,
            color=green_color,
            thickness=3
        )
        cv2.putText(
            rgb_image_with_ROI_definitions,
            str(ROI_name),
            top_left_corner,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=green_color,
            thickness=5,
            lineType=cv2.LINE_AA
        )

    return rgb_image_with_ROI_definitions


def get_ROIs_for_image(rgb_image, ROI_definitions):
    ROIs = {
        ROI_name: rgb.crop.crop_image(rgb_image, ROI_definition)
        for ROI_name, ROI_definition in ROI_definitions.items()
    }

    return ROIs
