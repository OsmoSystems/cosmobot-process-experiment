'''Utlities for analyzing colors'''

import numpy


def average_color_for_image(img):
    '''Calculate average color for entire image'''
    avg_color_per_row = numpy.average(img, axis=0)
    avg_color = numpy.average(avg_color_per_row, axis=0)
    return dict(
        r=avg_color[2],  # values come not in order
        g=avg_color[1],
        b=avg_color[0]
    )
