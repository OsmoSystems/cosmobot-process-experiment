#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='osmo_camera',
    version='0.0.1',
    author='Osmo Systems',
    author_email='dev@osmobot.com',
    description='Prototype code for an osmobot camera sensor',
    url='https://www.github.com/osmosystems/camera-sensor-prototype',
    packages=find_packages(),
    install_requires=[
        'boto',
        'ipywidgets',
        'numpy',
        'matplotlib',
        'opencv-python',
        'pandas',
        'Pillow',
        'plotly >= 3, < 4',
        'rawpy',
    ],
    include_package_data=True
)
