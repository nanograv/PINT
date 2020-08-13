"""PINT Is Not TEMPO3!"""

import numpy as np
import astropy
import astropy.constants as c
import astropy.time as time
import astropy.units as u
from astropy import log
from astropy.units import si
from pint.extern._version import get_versions
from pint.pulsar_ecliptic import PulsarEcliptic
from pint.pulsar_mjd import PulsarMJD
import pint.pulsar_mjd

__all__ = [
    "__version__",
    "ls",
    "dmu",
    "light_second_equivalency",
    "hourangle_second",
    "pulsar_mjd",
    "GMsun",
    "Tsun",
    "Tmercury",
    "Tvenus",
    "Tearth",
    "Tmars",
    "Tjupiter",
    "Tsaturn",
    "Turanus",
    "Tneptune",
    "J2000",
    "J2000ld",
    "JD_MJD",
    "pint_units",
    "PulsarEcliptic",
    "PulsarMJD",
]

__version__ = get_versions()["version"]

# Define a few important constants

# light-second unit
ls = u.def_unit("ls", c.c * 1.0 * u.s)

# DM unit (pc cm^-3)
dmu = u.def_unit("dmu", u.pc * u.cm ** -3)

# define equivalency for astropy units
light_second_equivalency = [(ls, si.second, lambda x: x, lambda x: x)]
# hourangle_second unit
hourangle_second = u.def_unit("hourangle_second", u.hourangle / np.longdouble(3600.0))

# Following are from here:
# http://ssd.jpl.nasa.gov/?constants (grabbed on 30 Dec 2013)
GMsun = 1.32712440018e20 * u.m ** 3 / u.s ** 2

# Solar mass in time units (sec)
Tsun = (GMsun / c.c ** 3).to(u.s)

# Planet system(!) masses in time units
Tmercury = Tsun / 6023600.0
Tvenus = Tsun / 408523.71
Tearth = Tsun / 328900.56  # Includes Moon!
Tmars = Tsun / 3098708.0
Tjupiter = Tsun / 1047.3486
Tsaturn = Tsun / 3497.898
Turanus = Tsun / 22902.98
Tneptune = Tsun / 19412.24

# The Epoch J2000
J2000 = time.Time("2000-01-01 12:00:00", scale="utc")
J2000ld = pint.pulsar_mjd.time_to_longdouble(J2000)
JD_MJD = 2400000.5
# PINT special units list
pint_units = {
    "H:M:S": u.hourangle,
    "D:M:S": u.deg,
    "lt-s": ls,
    "ls": ls,
    "Tsun": Tsun,
    "GMsun": GMsun,
    "MJD": u.day,
    "pulse phase": u.dimensionless_unscaled,
    "hourangle_second": hourangle_second,
}

import astropy.version

if astropy.version.major < 4:
    log.warning(
        "Using astropy version {}. To get most recent IERS data, upgrade to astropy >= 4.0".format(
            astropy.__version__
        )
    )
