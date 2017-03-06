from __future__ import print_function
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy, os, sys, hashlib, os.path
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
    src = ["pint/cutils/str2ld_py.pyx"]
else:
    print("Using existing 'C' source file...")
    src = ["pint/cutils/str2ld_py.c"]

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
# A list of the filenames
fnames = [os.path.split(x)[1] for x in data_urls]
# Following md5sums created using:
# >  dict([(fname, hashlib.md5(open(fname, 'rb').read()).digest()) for fname in fnames])
md5sums = {'de-403-masses.tpc': '\x00\x13{\xda\x957\xbbG\xcb\xd0v\xb6qoBg',
           'de405.bsp': '\xc5\xe5\x9d\x92\xfcO\x84`\xaa\xa1\xeeV\x94i0x',
           'de421.bsp': "a\x10\xd8\xd2\xb65`h\x0f\xf8\x0b'\xe3\x00e\xb2",
           'de430t.bsp': '\xe2p\xe5\xb9\xcc\x8c(\xa2\xd8#\x06\xbeLQU\xb6',
           'earth_latest_high_prec.bpc': '\xd3\xefHO%\x1dc\xb4\x91\x06\xc1\x93\xf6\x01\xbb\x08',
           'naif0012.tls': '%\xa2\xff\xf3\x0b\r\xed\xb4\xd7l\x06r{\x18\x95\xb1',
           'pck00010.tpc': '\xda\x156A\xf74k\xd5\xb6\xa1"gx\xe0\xd5\x1b'}
data_files = []
data_dir = 'datafiles'
for u in data_urls:
    fname = os.path.split(u)[1]
    # Download if not present or if md5sum doesn't match
    path = os.path.join("pint", data_dir, fname)
    if (not os.path.exists(path) or
        hashlib.md5(open(path, 'rb').read()).digest() != md5sums[fname]):
        os.system("wget -N -P pint/%s %s" % (data_dir, u))
    else:
        print("Downloaded file '%s' looks good." % fname)
    data_files.append(os.path.join(data_dir, fname))

# And now add the clock files (at least until we figure out a better
# way of doint this.  This aids in automatic testing, though.
clock_files = ['gps2utc.clk', 'time.dat',
               'time_jb.dat', 'time_nancay.dat', 'time_wsrt.dat',
               'time_gb853.dat', 'time_bonn.dat', 'time_vla.dat',
               'time_gb140.dat', 'time_gbt.dat', 'time_pks.dat',
               'time_ao.dat']
for fname in clock_files:
    data_files.append(os.path.join(data_dir, fname))

import versioneer
cmdclass.update(versioneer.get_cmdclass())

setup(
    name="pint",
    version = versioneer.get_version(),
    description = 'A Pulsar Timing Package, written in Python from scratch',

    author = 'Luo Jing, Scott Ransom, Paul Demorest, Paul Ray, et al.',
    author_email = 'sransom@nrao.edu',
    url = 'https://github.com/nanograv/PINT',
    license = 'TBD',

    install_requires = ['astropy>=1.2'],

    # These scripts will be installed in your PATH
    # These should be end user scripts for general use.
    scripts = ['scripts/pintempo.py', 'scripts/zima.py', 'scripts/pintbary.py', 'scripts/fermiphaseogram.py', 'scripts/nicerphase.py'],

    packages=['pint',
        'pint.extern',
        'pint.models',
        'pint.models.stand_alone_psr_binaries',
        'pint.observatory',
        'pint.orbital'],

    package_data={'pint': [
        'datafiles/ecliptic.dat', 
        'datafiles/gps2utc.clk'
        ] + data_files},

    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
