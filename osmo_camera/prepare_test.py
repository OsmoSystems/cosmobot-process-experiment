'''Test preparation for experiment'''
from prepare import is_hostname_valid
# from prepare import experiment_configuration, create_output_folder,
# from prepare import estimate_image_count, images_with_free_space, free_space_for_image_count
# from prepare import free_space_for_one_image, free_space_for_experiment, git_hash


# def experiment_configuration_test():
#     '''TODO:'''
#
#
# def create_output_folder_test():
#     '''TODO:'''


def is_hostname_valid_test():
    assert is_hostname_valid('pi-cam-2222') is True


def is_hostname_invalid_test():
    assert is_hostname_valid('pi-cam-2222') is False


# def estimate_image_count_test():
#     '''TODO:'''
#
#
# def images_with_free_space_test():
#     '''TODO:'''
#
#
# def free_space_for_image_count_test():
#     '''TODO:'''
#
#
# def free_space_for_one_image_test():
#     '''TODO:'''
#
#
# def free_space_for_experiment_test():
#     '''TODO:'''
#
#
# def git_hash_test():
#     '''TODO:'''
