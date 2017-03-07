# Top-level PINT __init__.py
"""
PINT Is Not TEMPO3!
"""

from .extern._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Define a few important constants
import astropy.units as u
from astropy.units import si
import astropy.constants as c
import astropy.time as time
from . import utils

# light-second unit
ls = u.def_unit('ls', c.c * 1.0 * u.s)

# DM unit (pc cm^-3)
dmu = u.def_unit('dmu', u.pc*u.cm**-3)

# define equivalency for astropy units
light_second_equivalency = [(ls, si.second, lambda x: x, lambda x: x)]

# Following are from here:
# http://ssd.jpl.nasa.gov/?constants (grabbed on 30 Dec 2013)
GMsun = 1.32712440018e20 * u.m**3/u.s**2

# Solar mass in time units (sec)
Tsun = (GMsun / c.c**3).to(u.s)

# Planet system(!) masses in time units
Tmercury = Tsun / 6023600.
Tvenus = Tsun / 408523.71
Tearth = Tsun / 328900.56 # Includes Moon!
Tmars = Tsun / 3098708.
Tjupiter = Tsun / 1047.3486
Tsaturn = Tsun / 3497.898
Turanus = Tsun / 22902.98
Tneptune = Tsun / 19412.24

# The Epoch J2000
J2000 = time.Time('2000-01-01 12:00:00', scale='utc')
J2000ld = utils.time_to_longdouble(J2000)
# PINT special units list
pint_units = {'H:M:S':u.hourangle,'D:M:S':u.deg,'lt-s':ls,'ls':ls,'Tsun':Tsun,
              'GMsun':GMsun,'MJD':u.day,'pulse phase':u.Unit(1)}
