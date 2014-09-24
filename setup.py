from distutils.core import setup
import numpy, os
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True
    
# The following needs to be set to point to the top-level directory
# of the PySPICE installation.
pyspice = "/home/sransom/git/PySPICE"

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
    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
