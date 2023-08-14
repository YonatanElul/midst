from distutils.core import setup
from setuptools import find_packages

import os

cwd = os.getcwd()

setup(
    name='midst',
    version='1.0',
    description=(
        'This is the official implementation of the code used in the paper '
        '"Data-Driven Modelling of Interrelated Dynamical Systems"'
    ),
    author='Yonatan Elul',
    author_email='johnneye@campus.technion.ac.il',
    url='https://github.com/YonatanElul/midst.git',
    license='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Private',
        'Topic :: Software Development :: Dynamical Systems Modelling',
        'Programming Language :: Python :: 3.8',
    ],
    package_dir={'midst': os.path.join(cwd, 'midst')},
    packages=find_packages(
        exclude=['data', 'logs'],
    ),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'torch',
        'torchvision',
        'tqdm',
        'h5py',
        'pandas',
        'netCDF4',
        'wfdb',
        'numba',
        'seaborn',
    ],
)
