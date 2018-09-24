import cv2

from osmo_camera import rgb


def choose_regions(rgb_image):
    ''' Funky interaction to select regions within an image.
    READ THIS:
    When you call this, the user must:
    1. go to the window that pops up
    2. click + drag to select a region
    3. PRESS ENTER ONCE. Pressing enter multiple times will save the same region again
    4. return to step 2 until you've selected all the regions you want
    5. after pressing enter the last time, close the window by pressing Esc a couple of times.

    Arguments:
        rbg_image: 3D numpy.ndarray as an RGB image
    Returns:
        numpy 2d array, essentially an iterable containing iterables of (start_col, start_row, cols, rows)
        corresponding to the regions that you selected.
    '''
    window_name = 'ROIs selection'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)  # WINDOW_GUI_EXPANDED seems to allow you to resize the window

    # Resize the window to a manageable default.
    window_size = 600  # in pixels
    cv2.resizeWindow(window_name, window_size, window_size)

    regions = cv2.selectROIs(window_name, rgb.convert.to_bgr(rgb_image))
    cv2.waitKey()

    cv2.destroyWindow(window_name)
    return regions


def prompt_for_ROI_selection(rgb_image):
    ROIs = choose_regions(rgb_image)

    ROI_definitions = {
        input(f'Name ROI Selection {index + 1}: '): list(ROI)  # Convert np array to list to make print readable
        for index, ROI in enumerate(ROIs)
    }

    return ROI_definitions
