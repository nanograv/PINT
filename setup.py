from distutils.core import setup
import numpy, os, sys
from distutils.extension import Extension
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
    print "Using cython..."
    src = ["spice_util_cython/spice_util_py.pyx"]
else:
    print "Using existing 'C' source file..."
    src = ["spice_util_cython/spice_util_py.c"]

ext_modules += [Extension("spice_util", src,
                          include_dirs = [numpy.get_include(),
                                          os.path.join(pyspice, "cspice/include")],
                          extra_objects = [os.path.join(pyspice, "cspice", "lib", "cspice.a")]),]

if use_cython:
    cmdclass.update({'build_ext': build_ext})

setup(
    name="pint",
    version = '0.0.1',
    description = 'A Pulsar Timing Package, written in Python from scratch',

    author = 'Luo Jing, Scott Ransom, et al.',
    author_email = 'sransom@nrao.edu',
    url = 'https://github.com/nanograv/PINT',

    packages=['pint', 
        'pint.models', 
        'pint.models.pulsar_binaries', 
        'pint.orbital'],

    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
