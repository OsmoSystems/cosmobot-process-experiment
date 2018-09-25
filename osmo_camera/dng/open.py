import rawpy


def as_rgb(dng_filename):
    ''' Opens a .DNG file as an `RGB image`, with absolutely minimal postprocessing. The only processing applied
    is the combination of the two green channels in the raw .DNG file into a single green channel

    Args:
        dng_filename: The name of the .DNG file to open

    Returns:
        An `RGB image`
    '''
    with rawpy.imread(dng_filename) as raw:
        # NOTE: to make this behave exactly the same as Jeremy's original code, we could set `four_color_rgb=True`,
        # which results in an RGBG image: a 2D numpy array of pixels, in which each pixel has a value for each of RGBG.
        # We could then manually remove the second green channel

        image = raw.postprocess(
            # These settings prevent any special post-processing from happening - pass through bayer colors directly
            half_size=True,  # Use one pixel per RGBG 2x2 instead of interpolating
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,  # Demosaic algorithm doesn't apply when half_size=True
            four_color_rgb=False,  # Combine the two green channels (somehow.. we don't quite know how)
            output_bps=16,  # Use 16-bit resolution
            gamma=(1, 1),  # Don't apply gamma-correction
            output_color=rawpy.ColorSpace.raw,  # Don't do any color conversion
            # Don't apply any auto white or brightness balancing
            use_camera_wb=True,
            no_auto_scale=True,
            no_auto_bright=True,
        )

    return image / 2 ** 16  # TODO (SOFT-514): remove magic number denormalizing, update everywhere to use 10-bit?
