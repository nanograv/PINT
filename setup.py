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
pyspice = "/YOUR/PATH/TO/PySPICE"

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

    packages=['pint'],
    package_dir = {'pint': 'pint'},

    py_modules = ['pint.models.timing_model', 'pint.models.astrometry',
        'pint.models.dispersion', 'pint.models.spindown',
        'pint.models.parameter', 'pint.fitter',
        'pint.ltinterface',
        'pint.models.solar_system_shapiro',
        'pint.models.btmodel', 'pint.models.polycos',
        'pint.models.bt', 'pint.models.dd',
        'pint.orbital.kepler'],

    include_package_data=True,
    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
