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
pyspice = "/home/sransom/src/PySPICE"

# The following is the command to use for building in-place for development
# python setup.py build_ext --inplace

# Check to see if we have a shared library for SPICE, yet
# With gcc, you need to do something like the following in
# pyspice/cspice/lib, and then copy libcspice.so to a directory
# in your LD_LIBRARY_PATH
# gcc -shared -o libcspice.so -L. -Wl,--whole-archive cspice.a -Wl,--no-whole-archive

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [Extension("spice_util_cython/spice_util_py",
                              ["spice_util_cython/spice_util_py.pyx"],
                            include_dirs = [numpy.get_include(),
                                os.path.join(pyspice, "cspice/include")],
                            libraries = ["cspice"],
                            library_dirs = [os.path.join(pyspice, "cspice/lib")]),]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [Extension("spice_util_cython/spice_util_py",
                              ["spice_util_cython/spice_util_py.c" ],
                            include_dirs = [numpy.get_include(),
                                os.path.join(pyspice, "cspice/include")],
                            libraries = ["cspice"],
                            library_dirs = [os.path.join(pyspice, "cspice/lib")]),]

setup(
    cmdclass = cmdclass,
    ext_modules=ext_modules,
)

