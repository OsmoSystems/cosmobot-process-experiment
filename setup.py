#!/usr/bin/env python
from setuptools import setup

setup(
    name='osmo_camera',
    version='0.0.1',
    author='Osmo Systems',
    author_email='dev@osmobot.com',
    description='Prototype code for an osmobot camera sensor',
    url='https://www.github.com/osmosystems/camera-sensor-prototype',
    packages=['osmo_camera'],
    install_requires=[
        'numpy',
        'opencv-python',
        'pandas',
        'plotly >= 2, < 3',
        'rawpy',
        'logging'
    ],
    include_package_data=True
)
