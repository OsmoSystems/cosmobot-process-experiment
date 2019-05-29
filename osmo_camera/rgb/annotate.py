from copy import deepcopy

import cv2


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


def draw_text_on_image(rgb_image, text):
    ''' Write the provided text on the provided image in the top left corner.

    Args:
        rgb_image: An RGB image
        text: String containing text to write on image
    Returns:
        A new RGB image
    '''
    rgb_image_with_timestamp = deepcopy(rgb_image)

    text_color_rgb = (0, 1, 0)
    text_position = (20, 50)
    font_scale = 1.5

    cv2.putText(
        rgb_image_with_timestamp,
        text,
        text_position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=text_color_rgb,
        thickness=5,
        lineType=cv2.LINE_AA
    )

    return rgb_image_with_timestamp
