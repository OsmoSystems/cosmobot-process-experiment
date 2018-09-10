'''Process experiment results'''
import os
from subprocess import call


def convert_to_dng(input_file, output_file):
    '''Perform conversion of jpeg+exif to dng'''
    # Requires that raspi_dng is made using the makefile from and copied to /usr/local/bin
    # 1) git clone https://github.com/illes/raspiraw
    # 2) cd ~/raspiraw-master && make
    # 3) sudo cp ~/raspiraw-master/raspi_dng /user/local/bin
    comm = 'raspi_dng {} {}'.format(input_file, output_file)
    print("Converting jpeg to dng: {}".format(comm))
    call([comm], shell=True)


def convert_img_in_dir_to_dng(directory):
    '''Convert all jpegs in a directory to dng'''
    for _, _, files in os.walk(directory):
        for input_file in files:
            filename, _ = os.path.splitext(input_file)
            output_file = directory + '/' + filename + '.dng'
            convert_to_dng(directory + '/' + input_file, output_file)


def s3_sync_output_dir(directory='./output'):
    '''Runs aws s3 sync command with output folder'''
    # Using CLI vs boto: https://github.com/boto/boto3/issues/358
    # It looks like sync is not a supported function of the python boto library
    # Work around is to use cli sync for now (requires aws cli to be installed)
    print("Performing sync of output (experiments) folder")

    # This argument pattern issues a uni-directional sync to S3 bucket
    # https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html
    comm = 'aws s3 sync {} s3://camera-sensor-experiments'.format(directory)
    call([comm], shell=True)
