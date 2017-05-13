#!/bin/python
from setuptools import setup, Extension
from setuptools import find_packages
from codecs import open
from os import path

__version__ = '0.0.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='snn1337',
    version=__version__,
    description='spiking neural network framework ',
    long_description=long_description,
    url='https://github.com/demikandr/snn1337',
    download_url='https://github.com/demikandr/snn1337/tarball/' + __version__,
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 2'
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='snn1337 contributors',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='',
    ext_modules=[Extension(
        "snn1337.neuron",
        sources=["snn1337/cython/neuron.cpp"],
        language="c++"
    )]
)
