'''d'''
import json
from cv2 import Canny, findContours, boundingRect, imwrite, RETR_LIST, CHAIN_APPROX_SIMPLE
from color import average_color_for_image


def extract_patches_from_image(image, node_id, output_file_prefix):
    '''d'''
    edged = Canny(image, 10, 250)
    (_, contours_detected, _) = findContours(edged.copy(), RETR_LIST, CHAIN_APPROX_SIMPLE)
    idx = 0

    results_dict = dict()
    used_contour_dict = dict()

    for contour in contours_detected:
        x_coor, y_coor, width, height = boundingRect(contour)

        if (width > 50 and height > 50) and (width < 200 and height < 200):
            used_contour_dict_key = '{}{}{}{}'.format(x_coor, y_coor, width, height)

            if used_contour_dict_key not in used_contour_dict:
                used_contour_dict[used_contour_dict_key] = True

                results_dict_key = 'patch{}'.format(idx)
                idx += 1
                new_img = image[y_coor:y_coor+height, x_coor:x_coor+width]
                avg_color = average_color_for_image(new_img)
                image_filename = "./output/{}_node_{}_patch_{}.png".format(output_file_prefix, node_id, idx)
                imwrite(image_filename, new_img)

                results_dict[results_dict_key] = dict()
                results_dict[results_dict_key]["contour"] = dict(
                    x=x_coor,
                    y=y_coor,
                    width=width,
                    height=height
                )
                results_dict[results_dict_key]["avg_color"] = avg_color

    with open('./output/{}_node{}data.json'.format(output_file_prefix, node_id), 'w') as outfile:
        json.dump(results_dict, outfile)
