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


def capture_iterating_ss_and_iso(output_folder, base_filename, start_ss=500000, end_ss=1000000, step_ss=500000,
                                 start_iso=100, end_iso=200, step_iso=100):
    ''' Capture multiple images iterating through a list of shutter speeds
        and isos

    Args:
        output_folder: folder to write file to
        base_filename: filename to write file to
        start_ss: first shutter speed to use for capture
        end_ss: last shutter speed value to use for capture
        step_ss: step value for shutter speed
        start_iso: first iso to use for capture
        end_iso: final iso value to use for capture
        step_iso: step value for iso

    Returns:
        None
    '''
    for s_s in range(start_ss, end_ss + 1, step_ss):
        for iso in range(start_iso, end_iso + 1, step_iso):
            filename = output_folder + f"/{base_filename}_ss_{s_s}_iso_{iso}.jpeg"
            capture(filename, '--ex {} --ISO {}'.format(s_s, iso))
