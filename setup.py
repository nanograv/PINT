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
pyspice = None
for arg in sys.argv:
    if arg.startswith('--with-pyspice='):
        pyspice = arg.split('=', 1)[1]
    else:
        argv_replace.append(arg)
sys.argv = argv_replace

if pyspice is None and 'PYSPICE' in os.environ:
    # We have the PYSPICE directory set as an environment variable
    pyspice = os.environ['PYSPICE']

if pyspice is None:
    # We need to auto-detect the directory of PySPICE in some way here.
    # TODO: create some auto-detect code here

    # For now, set pyspice manually here
    # The following needs to be set to point to the top-level directory
    # of the PySPICE installation.
    pyspice = "/YOUR/PATH/TO/PySPICE"

    print("""
PINT was unable to autodetect the location of the PySPICE source. Using the
default location from the setup script. If you get errors, please run setup.py
again, but use the option --with-pyspice=.. to point PINT to the PySPICE source
directory (e.g., /home/username/code/PySPICE/)
""")

print("Using PySPICE directory: {0}".format(pyspice))

# The following is the command to use for building in-place for development
# python setup.py build_ext --inplace

cmdclass = {}
ext_modules = []

if use_cython:
    print("Using cython...")
    src = ["spice_util_cython/spice_util_py.pyx"]
else:
    print("Using existing 'C' source file...")
    src = ["spice_util_cython/spice_util_py.c"]

ext_modules += [Extension("spice_util", src,
                          include_dirs = [numpy.get_include(),
                                          os.path.join(pyspice, "cspice/include")],
                          extra_objects = [os.path.join(pyspice, "cspice", "lib", "cspice.a")]),]

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
