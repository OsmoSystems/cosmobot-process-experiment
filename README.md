# Terminology
* `RAW image file` - A JPEG+RAW image file as directly captured by a PiCam v2, saved as a .JPEG
* `.DNG image file` - An RAW image file converted to the Adobe Digital Negative (.DNG) format
* `RGB image` - A 3D numpy.ndarray: a 2D array of "pixels" (row-major), where each "pixel" is a 1D array of [red, green, blue] channels with a value between 0 and 1. This is our default format for interacting with images. An example 4-pixel (2x2) image would have this shape:

```
[
 [ [r1, g1, b1], [r2, g2, b2] ],
 [ [r3, g3, b3], [r4, g4, b4] ]
]
```

* `BGR image` - A 3D numpy.ndarray: a 2D array of "pixels", where each "pixel" is a 1D array of [blue, green, red] channels with a value between 0 and 1. This is OpenCV's default
* `ROI` - A rectangular Region of Interest (ROI) in a given image.
* `ROI definition` - A 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows), used to define a Region of Interest (ROI).


# Usage
The function `process_experiment` is meant to be run from a local Jupyter Notebook.

## Pre-reqs
Some set up is required before you can run `process_experiment` locally. These directions are very minimal - check with software team if you have questions or run into problems.

### awscli
Install:
1. Follow instructions here: https://docs.aws.amazon.com/cli/latest/userguide/installing.html
Note: I had to use `pip` and not `pip3` to successfully install awscli
2. Run `aws --version` from the command line to confirm it's installed correctly

Set up credentials:
1. Generate an AWS access key
2. Run `aws configure` and put in the access key ID and secret key
3. Confirm your credentials are working by running `aws s3 ls` - you should get a list of folders

### raspiraw
1. Clone our forked repo:
```
git clone https://github.com/OsmoSystems/raspiraw.git
```

2. Navigate to that directory and run `make`, e.g.:
```
~$ cd ~/osmo/raspiraw
~/osmo/raspiraw$ make
```

## Running
See this example jupyter notebook: "2018-09-25 process_experiment example snippet.ipynb" in https://drive.google.com/drive/folders/1bEVTuDQf_Ej_Ct-iggXkkg1NOTVrGHXc
