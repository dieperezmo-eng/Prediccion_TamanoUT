"""Package installer"""
# docstring for the setup module. This module is responsible for setting up the package installation.
# It uses setuptools to define the package metadata and dependencies.
# setup.py is a standard file used in Python projects to facilitate package distribution.

from setuptools import find_packages, setup  # type: ignore

setup(
    name="homework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "pandas",
        "scikit-learn",
        "ipykernel",
        "matplotlib",
        "seaborn",
        "numpy",
    ],
)
