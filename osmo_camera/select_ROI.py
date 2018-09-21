import cv2


def choose_regions(image):
    ''' Funky interaction to select regions within an image.
    READ THIS:
    When you call this, the user must:
    1. go to the window that pops up
    2. click + drag to select a region
    3. PRESS ENTER ONCE. Pressing enter multiple times will save the same region again
    4. return to step 2 until you've selected all the regions you want
    5. after pressing enter the last time, close the window by pressing Esc a couple of times.

    Arguments:
        image: numpy.ndarray of an openCV-style image
    Returns:
        numpy 2d array, essentially an iterable containing iterables of (start_col, start_row, cols, rows)
        corresponding to the regions that you selected.
    '''
    window_name = 'ROIs selection'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)  # WINDOW_GUI_EXPANDED seems to allow you to resize the window

    # Resize the window to a manageable default.
    window_size = 600  # in pixels
    cv2.resizeWindow(window_name, window_size, window_size)

    regions = cv2.selectROIs(window_name, image)
    cv2.waitKey()

    cv2.destroyWindow(window_name)
    return regions


def prompt_for_ROI_selection(image):
    # TODO: finish implementing, test
    # Require each ROI to be labelled
    # Make it easier to label "high" and "low"?
    return choose_regions(image)
