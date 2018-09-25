#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='osmo_camera',
    version='0.0.1',
    author='Osmo Systems',
    author_email='dev@osmobot.com',
    description='Prototype code for an osmobot camera sensor',
    url='https://www.github.com/osmosystems/camera-sensor',
    packages=find_packages(),
    install_requires=[
        'boto',
        'numpy',
        'exifread',
        'opencv-python',
        'pandas',
        'plotly >= 2, < 3',
        'rawpy',
    ],
    include_package_data=True
)
