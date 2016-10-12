from __future__ import print_function
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy, os, sys
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

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
    src = ["pint/str2ld/str2ld_py.pyx"]
else:
    print("Using existing 'C' source file...")
    src = ["pint/str2ld/str2ld_py.c"]

ext_modules += [Extension("pint.str2ld", src,
                          include_dirs = [numpy.get_include(),],
                          )]

if use_cython:
    cmdclass.update({'build_ext': build_ext})

# Download data files
data_urls = [
        "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de405.bsp",
        "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de421.bsp",
        "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de430t.bsp",
        "http://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
        "http://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
        "http://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        "http://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/de-403-masses.tpc"
        ]
data_files = []
data_dir = 'datafiles'
for u in data_urls:
    os.system("wget -N -c -P pint/%s %s" % (data_dir, u))
    data_files.append(os.path.join(data_dir,u.split('/')[-1]))

setup(
    name="pint",
    version = '0.0.1',
    description = 'A Pulsar Timing Package, written in Python from scratch',

    author = 'Luo Jing, Scott Ransom, et al.',
    author_email = 'sransom@nrao.edu',
    url = 'https://github.com/nanograv/PINT',


    packages=['pint',
        'pint.extern',
        'pint.models',
        'pint.models.stand_alone_psr_binaries',
        'pint.orbital'],

    package_data={'pint':['datafiles/observatories.txt',
                          'datafiles/ecliptic.dat', ]+data_files},

    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
