from subprocess import check_output


def capture(filename, additional_capture_params=''):
    ''' Capture raw image JPEG+EXIF using command line

    Args:
        filename: filename to save an image to
        additional_capture_params: Additional parameters to pass to raspistill command

    Returns:
        Resulting command line output of the raspistill command
    '''
    command = f'raspistill --raw -o {filename} {additional_capture_params}'
    print(f'Capturing image using raspistill: {command}')
    command_output = check_output(command, shell=True).decode("utf-8")
    return command_output


def simulate_capture_with_copy(filename):
    ''' Simulate capture by copying image file

    Args:
        filename: filename to copy a test image to

    Returns:
        Resulting command line output of the copy command
    '''
    command = f'cp ./image_for_development.jpeg {filename}'
    print(f'Emulate capture using cp: {command}')
    command_output = check_output(command, shell=True).decode("utf-8")
    return command_output
