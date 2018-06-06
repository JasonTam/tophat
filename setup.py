#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Note: To use the 'upload' functionality of this file, you must:
    $ pip install twine

References:
    https://github.com/kennethreitz/setup.py
"""

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'top-hat'
DESCRIPTION = 'Recommendation system in TensorFlow'
URL = 'https://github.com/gilt/tophat'
EMAIL = 'cerebro-dev@gilt.com (jtam@gilt.com)'
AUTHOR = 'GILT-cerebro (Jason Tam)'
REQUIRES_PYTHON = '>=3.6.0'
LICENSE = 'MIT'  # NOTE: update classifiers if modified
VERSION = '0.0.1'

# What packages are required for this module to be executed?
EXTRAS_REQUIRES = {
    'tf': ['tensorflow>=1.6.0'],
    'tf_gpu': ['tensorflow-gpu>=1.6.0'],
}

INSTALL_REQUIRES = [
   'pandas', 'numpy', 'scipy', 'tqdm'
]

TESTS_REQUIRES = [
    'pytest', 'requests',
]

SETUP_REQUIRES = [
    'pytest-runner',        
]

# -----------------------------------------------------------------------------

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(
            sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    extras_require=EXTRAS_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRES,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)

