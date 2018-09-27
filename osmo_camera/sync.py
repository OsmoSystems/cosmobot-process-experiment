'''DOCSTRING:'''

from multiprocessing import Process
from subprocess import call

# global
SYNC_PROCESSES = dict()


def sync_with_s3(directory='./output'):
    '''sync a directory with s3://camera-sensor-experiments using an external
       shell command.
     Args:
        directory: directory to sync
     Returns:
        None
    '''

    # Using CLI vs boto: https://github.com/boto/boto3/issues/358
    # It looks like sync is not a supported function of the python boto library
    # Work around is to use cli sync for now (requires aws cli to be installed)
    print("Performing sync of output (experiments) folder")

    # This argument pattern issues a uni-directional sync to S3 bucket
    # https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html
    comm = 'aws s3 sync {} s3://camera-sensor-experiments'.format(directory)
    call([comm], shell=True)


def sync_for_directory_is_alive(directory_key):
    '''Checks if a sync process is currently being run for the directory passed in
     Args:
        directory_key: key to check within the SYNC_PROCESSES dictionary
     Returns:
        Boolean: Used to check if a directory is currently syncing
    '''
    if directory_key in SYNC_PROCESSES and SYNC_PROCESSES[directory_key].is_alive():
        return True

    return False


def end_syncing_processes():
    '''Stop all processes currently syncing.  Intended to be used
       if experimental image capture has finished and a final
       sync should be initiated.
     Args:
        None
     Returns:
        None
    '''

    for directory_key in SYNC_PROCESSES:
        process = SYNC_PROCESSES[directory_key]
        process.stop()
        SYNC_PROCESSES[directory_key] = None


# TODO: Process limiting?  Fast-follow ticket using a queue?'''
# TODO: Test in real life: Will process management algorithm cause inconsitencies with syncing'''

def sync_directory_in_separate_process(directory, final_sync=False):
    '''Instantiates a separate process for syncing a directory.  Stores
       a reference to the process to check later for subsequent syncs.
     Args:
        directory: directory to sync
        final_sync (optional): Should the newly invoked process be completed before
        returning from the function (async/sync).
     Returns:
        None.
    '''
    if sync_for_directory_is_alive(directory):
        return

    sync_process = Process(target=sync_with_s3, args=(directory,))
    sync_process.start()
    SYNC_PROCESSES[directory] = sync_process

    if final_sync:
        end_syncing_processes()
        sync_process.join()
