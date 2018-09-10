'''Process experiment results'''
import os
from subprocess import call

def convert_to_dng(input_file, output_file):
    '''Perform conversion of jpeg+exif to dng'''
    comm = 'raspi_dng {} {}'.format(input_file, output_file)
    print("Converting jpeg to dng: {}".format(comm))
    call([comm], shell=True)

def convert_img_in_dir_to_dng(directory):
    '''Convert all jpegs in a directory to dng'''
    for _, _, files in os.walk(directory):
        for input_file in files:
            filename, _ = os.path.splitext(input_file)
            output_file = directory + filename + '.dng'
            convert_to_dng(directory + input_file, output_file)

def s3_sync_output_dir():
    '''Runs aws s3 sync command with output folder'''
    # Using CLI vs boto: https://github.com/boto/boto3/issues/358
    # It looks like sync is not a supported function of the python boto library
    # Work around is to use cli sync for now (requires aws cli to be installed)
    print("Performing sync of output (experiments) folder")
    comm = 'aws s3 sync ./output/ s3://camera-sensor-experiments'
    call([comm], shell=True)
