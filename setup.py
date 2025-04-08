# import os
import subprocess

# Ensure required dependencies are installed before proceeding
try:
    from setuptools import setup, find_packages
except ImportError:
    subprocess.check_call(["pip", "install", "setuptools"])
    from setuptools import setup, find_packages

def read_requirements():
    try:
        with open('requirements.txt') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError("Error: 'requirements.txt' not found. Please ensure the file exists in the project directory.")

# General setup file
setup(
    name='deep',
    version='0.1',
    description='A Deep Learning Framework for Animal Vision.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='AI',
    author_email='r2015469@novaims.unl.pt',
    url='https://github.com/LeonardoDiCaterina/DL',
    license='MIT',
    packages=find_packages(where='deep'),
    package_dir={'': 'deep'},
    install_requires=read_requirements(),
    python_requires='==3.11.11'
)
