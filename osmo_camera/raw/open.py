from picamraw import PiRawBayer, PiCameraVersion


def as_rgb(raw_image_path):
    raw_bayer = PiRawBayer(
        filepath=raw_image_path,
        camera_version=PiCameraVersion.V2,
        sensor_mode=0
    )

    # Divide by the bit-depth of the raw data to normalize into the (0,1) range
    RAW_BIT_DEPTH = 2**10
    rgb_image = raw_bayer.to_rgb() / RAW_BIT_DEPTH

    return rgb_image
