from __future__ import print_function
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

import binascii
import os
import sys
import hashlib
import os.path
# !! This means setup.py can't even be run without numpy installed!
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

import versioneer

# We need to find PySPICE. Can be done in one of two ways:
# Set the environment variable $PYSPICE, or use
# --with-pyspice=/direcory/to/pyspice
argv_replace = []
for arg in sys.argv:
    argv_replace.append(arg)
sys.argv = argv_replace

# The following is the command to use for building in-place for development
# python setup.py build_ext --inplace

cmdclass = {}
ext_modules = []

if use_cython:
    print("Using cython...")
    src = ["pint/cutils/str2ld_py.pyx"]
else:
    print("Using existing 'C' source file...")
    src = ["pint/cutils/str2ld_py.c"]

ext_modules += [Extension("pint.str2ld", src,
                          include_dirs = [numpy.get_include(),],
                          )]

if use_cython:
    cmdclass.update({'build_ext': build_ext})


# Make sure data files are installed.
def hex_hash(path):
    h_f = hashlib.md5(open(path, 'rb').read()).digest()
    return binascii.hexlify(h_f).decode('ascii')


data_files = []
data_dir = 'datafiles'

# And now add the clock files (at least until we figure out a better
# way of doint this.  This aids in automatic testing, though.
clock_files = ['gps2utc.clk', 'time.dat', 'tai2tt_bipm2015.clk',
               'time_jb.dat', 'time_nancay.dat', 'time_wsrt.dat',
               'time_gb853.dat', 'time_bonn.dat', 'time_vla.dat',
               'time_gb140.dat', 'time_gbt.dat', 'time_pks.dat',
               'time_ao.dat', 'time_hobart.dat', 'time_chime.dat']
for fname in clock_files:
    data_files.append(os.path.join(data_dir, fname))

cmdclass.update(versioneer.get_cmdclass())

# These command-line scripts will be built by the setup process and installed in your PATH
# See http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point
console_scripts = [ 'photonphase=pint.scripts.photonphase:main',
                    'event_optimize=pint.scripts.event_optimize:main',
                    'event_optimize_multiple=pint.scripts.event_optimize_multiple:main',
                    'pintempo=pint.scripts.pintempo:main',
                    'zima=pint.scripts.zima:main',
                    'pintbary=pint.scripts.pintbary:main',
                    'fermiphase=pint.scripts.fermiphase:main',
                    'pintk=pint.scripts.pintk:main' ]

setup(
    name="pint",
    version = versioneer.get_version(),
    description = 'A Pulsar Timing Package, written in Python from scratch',

    author = 'Luo Jing, Scott Ransom, Paul Demorest, Paul Ray, et al.',
    author_email = 'sransom@nrao.edu',
    url = 'https://github.com/nanograv/PINT',
    license = 'TBD',

    install_requires = ['astropy>=2.0'],
    setup_requires = ['pytest-runner>=2.0,<3dev'],
    tests_require = ['pytest', 'sphinx>=2.2.0'],

    entry_points={
        'console_scripts': console_scripts,
    },

    packages=['pint',
        'pint.extern',
        'pint.models',
        'pint.scripts',
        'pint.pintk',
        'pint.models.stand_alone_psr_binaries',
        'pint.observatory',
        'pint.orbital',
        'pint.templates'],

    package_data={'pint': [
        'datafiles/ecliptic.dat', # for ecliptic coordinates
        'datafiles/de432s.bsp',   # for testing purposes
        ] + data_files},

    cmdclass = cmdclass,
    ext_modules=ext_modules,
    #test_suite='tests',
    #tests_require=[]
)
