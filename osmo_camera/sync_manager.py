import multiprocessing
from s3 import sync_to_s3

# sync processes keyed by the directory to be synced
_SYNC_PROCESSES = dict()


def _sync_for_directory_is_alive(directory_key):
    '''Checks if a sync process is currently being run for the directory passed in
     Args:
        directory_key: key to check within the SYNC_PROCESSES dictionary
     Returns:
        Boolean: Used to check if a directory is currently syncing
    '''
    return (directory_key in _SYNC_PROCESSES and _SYNC_PROCESSES[directory_key] is not None and
            _SYNC_PROCESSES[directory_key].is_alive())


def end_syncing_processes():
    '''Stop all processes currently syncing.  Intended to be used
       if experimental image capture has finished and a final
       sync should be initiated.
     Args:
        None
     Returns:
        None
    '''

    for directory_key in _SYNC_PROCESSES:
        process = _SYNC_PROCESSES[directory_key]
        process.terminate()
        _SYNC_PROCESSES[directory_key] = None


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
    if _sync_for_directory_is_alive(directory):
        return

    sync_process = multiprocessing.Process(target=sync_to_s3, args=(directory,))
    sync_process.start()
    _SYNC_PROCESSES[directory] = sync_process

    if final_sync:
        sync_process.join()
