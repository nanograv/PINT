from distutils.core import setup
import numpy
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

extensions = [
    Extension("spkezr_array", ["spice_util_py.pyx"],
    include_dirs = [numpy.get_include(),
                "/Users/jingluo/Research_codes/pySPICE/PySPICE/cspice/include /Users/jingluo/Research_codes/pySPICE/PySPICE/cspice/lib/cspice.a",],
    libraries = ["cspice.a"],
    library_dirs = ["/Users/jingluo/Research_codes/pySPICE/PySPICE/cspice/lib"]
    ),

]

setup(
    cmdclass={'build_ext':build_ext},
    ext_modules= cythonize(extensions),
)                 
