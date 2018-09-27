'''import logging module and sys module for python logging and command line argument parsing'''
import logging
import os

# path to output to log file.  We use a relative path from this file (log.py) as importing
# the log module from other directories will break explicit relative paths
# e.g. './log/hub.log' is relative to the file that is importing log log module
LOG_PATH = os.path.dirname(os.path.realpath(__file__)) + '/log/hub.log'

# format of output message to be logged
LOG_FORMAT = '%(levelname)s %(asctime)s: %(message)s'

# log level for hub application to use for output
LOG_LEVEL = logging.DEBUG

# handlers for streaming output to console or to a file
HANDLERS = [logging.FileHandler(LOG_PATH), logging.StreamHandler()]

# configure logging library with handlers and format
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=HANDLERS)

DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def log(log_level=logging.INFO, msg="No log message provided"):
    '''hi'''
    logging.log(log_level, msg)
