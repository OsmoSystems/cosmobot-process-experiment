'''Perform camera capture experiment'''

from socket import gethostname
import re

def is_hostname_valid():
    '''Check to see if the hostname on the pi is valid and quit if it is not'''
    # confirm that the hostname conatins 'pi_cam' and that the last 4 chars of the hostname
    # are similar to the last 4 chars of a mac address
    if re.search("[0-9]{4}$", gethostname()) and re.search("pi_cam", gethostname()):
        quit()
