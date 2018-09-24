def to_bgr(rgb_image):
    ''' Converts an `RGB image` to a `BGR image`

    Args:
        rgb_image: An `RGB image`

    Returns:
        A `BGR image`
    '''

    # TODO: figure out why this magically works??
    # https://www.scivision.co/numpy-image-bgr-to-rgb/
    bgr_image = rgb_image[..., :: -1]

    return bgr_image
