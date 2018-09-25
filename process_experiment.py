'''Process experiment results'''
import os
from subprocess import call, check_output

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

def is_repo_present():
    return os.path.exists('./.gitignore') # TODO: better check?

def get_git_hash():
    comm = 'git rev-parse HEAD'
    comm_output = check_output(comm, shell=True).decode("utf-8")
    print(comm_output)
