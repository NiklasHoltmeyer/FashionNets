# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

# with open('LICENSE') as f:
#   license = f.read()
#
# with open('README.rst') as f:
#   readme = f.read()

setup(
    name='fashion_nets',
    version='0.1.0',
    description='',
    # long_description=readme,
    author='Niklas Holtmeyer',
    url='https://github.com/NiklasHoltmeyer/FashionNets',
    # license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=required,
)
