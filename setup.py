# File: setup.py
# File Created: Friday, 24th December 2021 12:58:29 pm
# Author: Steven Atkinson (steven@atkinson.mn)

from setuptools import setup, find_packages

requirements = [
    "scipy"
]

setup(
    name="deeponet_data",
    version="0.0.0",
    description="DeepONet data",
    author="Steven Atkinson",
    author_email="steven@atkinson.mn",
    url="https://github.com/sdatkinson/deeponet",
    install_requires=requirements,
    packages=find_packages(),
)
