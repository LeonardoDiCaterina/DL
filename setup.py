# import os
import subprocess

# Ensure required dependencies are installed before proceeding
try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    subprocess.check_call(["pip", "install", "setuptools"])
    from setuptools import setup, find_packages, Extension  # Retry import

# try:
#     from Cython.Build import cythonize
# except ImportError:
#     subprocess.check_call(["pip", "install", "Cython"])
#     from Cython.Build import cythonize  # Retry import


# # Finds all instances of '.pyx' files inside 'cifo'
# def find_pyx_files(package_dir="cifo"):
#     """Find all .pyx files in the package and return them as Extension modules."""
#     pyx_files = []
#     for root, _, files in os.walk(package_dir):
#         for file in files:
#             if file.endswith(".pyx"):
#                 module_path = os.path.splitext(os.path.relpath(os.path.join(root, file), package_dir))[0]
#                 module_name = module_path.replace(os.path.sep, ".")  # Convert to module format
#                 pyx_files.append(Extension(f"cifo.{module_name}", sources=[os.path.join(root, file)]))
#     return pyx_files


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
   # ext_modules=cythonize(find_pyx_files("deep"))
)
