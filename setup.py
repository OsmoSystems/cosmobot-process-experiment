#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='osmo_camera',
    version='0.0.1',
    author='Osmo Systems',
    author_email='dev@osmobot.com',
    description='Code for processing image-collection experiments',
    url='https://www.github.com/osmosystems/cosmobot-process-experiment.git',
    packages=find_packages(),
    entry_points={},
    install_requires=[
        'boto',
        'ipywidgets',
        'numpy',
        'matplotlib',
        'opencv-python',
        'pandas',
        'picamraw',
        'Pillow',
        'plotly >= 3, < 4',
        'scipy',
        'tifffile == 2018.11.28',
        'tqdm',
    ],
    include_package_data=True
)
