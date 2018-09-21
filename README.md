camerasensor


# Terminology
`RAW image file` - A JPEG+RAW image file as directly captured by a PiCam v2, saved as a .JPEG
`DNG image file` - An RAW image file converted to the Adobe Digital Negative (.DNG) format

`RGB image` - A 3D numpy.ndarray: a 2D array of "pixels" (row-major), where each "pixel" is a 1D array of [red, green, blue] channels. This is our default format for interacting with images. An example 4-pixel (2x2) image would have this shape:

[
 [ [r1, g1, b1], [r2, g2, b2] ],
 [ [r3, g3, b3], [r4, g4, b4] ]
]

`BGR image` - A 3D numpy.ndarray: a 2D array of "pixels", where each "pixel" is a 1D array of [blue, green, red] channels. This is OpenCV's default

`ROI` - A rectangular Region of Interest (ROI) in a given image.
`ROI definition` - A 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows), used to define a Region of Interest (ROI).
