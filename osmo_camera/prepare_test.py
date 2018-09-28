'''Test preparation for experiment'''
from . import prepare
# from prepare import experiment_configuration, create_output_folder,
# from prepare import estimate_image_count, images_with_free_space, free_space_for_image_count
# from prepare import free_space_for_one_image, free_space_for_experiment, git_hash


# def experiment_configuration_test():
#     '''TODO:'''
#
#
# def create_output_folder_test():
#     '''TODO:'''


def test_is_hostname_valid():
    assert prepare.is_hostname_valid('pi-cam-2222') is True


def test_is_hostname_invalid():
    assert prepare.is_hostname_valid('I am a naughty hostname') is False


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
