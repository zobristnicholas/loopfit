import os
import sys
import subprocess
from skbuild import setup
from setuptools import find_packages


def get_version(path):
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                sep = '"' if '"' in line else "'"
                return line.split(sep)[1]
        else:
            raise RuntimeError("Unable to find version string.")


if "clean" not in sys.argv:  # don't run if already cleaning
    # clean the build directory before running to avoid build errors when developing
    subprocess.check_call(["python", os.path.abspath(__file__), "clean"])


setup(
    name='loopfit',
    description='Superconducting resonator IQ loop fitting optimized for speed',
    version=get_version("src/loopfit/__init__.py"),
    author='Nicholas Zobrist',
    license='GPLv3',
    url='http://github.com/zobristnicholas/loopfit',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy']
)
