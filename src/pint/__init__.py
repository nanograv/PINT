"""PINT Is Not TEMPO3!

This package has many submodules, but useful starting places may be
:class:`pint.toa.TOAs`, :class:`pint.models.timing_model.TimingModel`, and
:class:`pint.residuals.Residuals`.

Below you will find a tree of submodules. The online documentation should also
provide a usable table of contents.

These docstrings contain reference documentation; for tutorials, explanations,
or how-to documentation, please see other sections of the online documentation.
"""

import os
import re
import sys
import warnings

import astropy
import astropy.constants as c
import astropy.time as time
import astropy.units as u
import numpy as np
import pkg_resources
from astropy.units import si
from loguru import logger as log

from pint.extern._version import get_versions
from pint.pulsar_ecliptic import PulsarEcliptic
from pint.pulsar_mjd import PulsarMJD, time_to_longdouble  # ensure always loaded

"""
Want loguru to capture warnings emitted by warnings.warn
See https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings
"""
warn_ = warnings.warn


def warn(message, *args, **kwargs):
    if len(args) > 0:
        log.warning(f"{args[0]} {message}")
    elif "category" in kwargs:
        log.warning(f"{kwargs['category']} {message}")
    else:
        log.warning(f"{message}")
    warn_(message, *args, **kwargs)


warnings.warn = warn


class LogFilter:
    """Custom logging filter for loguru.
    Define some messages that are never seen (e.g., Deprecation Warnings).
    Others that will only be seen once.  Filtering of those is done on the basis of regex"""

    def __init__(self, onlyonce=None, never=None):
        # Define regexs for messages that will only be seen once.  Use "\S+" for a variable that might change
        # If a message comes through with a new value for that variable, it will be seen
        # Make sure to escape other regex commands like ()
        # Each message starts with state = False
        # Once it has been emitted, that changes to a list of the messages so that it can keep track
        # These are only suppressed when at level INFO or lower
        self.onlyonce = {
            "Using EPHEM = \S+ for \S+ calculation": False,
            "Using CLOCK = \S+ from the given model": False,
            "Using PLANET_SHAPIRO = \S+ from the given model": False,
            "Applying clock corrections \(include_gps = \S+, include_bipm = \S+\)": False,
            "Applying observatory clock corrections.": False,
            "Applying GPS to UTC clock correction \(\~few nanoseconds\)": False,
            "Computing \S+ columns.": False,
            "Using EPHEM = \S+ for \S+ calculation.": False,
            "Planet PosVels will be calculated.": False,
            "Computing PosVels of observatories, Earth and planets, using \S+": False,
            "Set solar system ephemeris to \S+": False,
            "Adding column \S+": False,
            "Adding columns .*": False,
            "Applying TT\(\S+\) to TT\(\S+\) clock correction \(\~27 us\)": False,
            "No pulse number flags found in the TOAs": False,
        }
        # add in any more defined on init
        if onlyonce is not None:
            for m in onlyonce:
                self.onlyonce[m] = False
        # List of matching strings for messages never to be displayed
        self.never = ["MatplotlibDeprecationWarning", "DeprecationWarning"]
        # add in any more defined on init
        if never is not None:
            self.never += never

    def filter(self, record):
        """Filter the record based on record["message"] and record["level"]
        If this returns False, the message is not seen
        """
        for m in self.never:
            if m in record["message"]:
                return False
        # display all warnings and above
        if record["level"].no >= log.level("WARNING").no:
            return True
        for m in self.onlyonce:
            if re.match(m, record["message"]):
                if not self.onlyonce[m]:
                    self.onlyonce[m] = [record["message"]]
                    return True
                elif not (record["message"] in self.onlyonce[m]):
                    self.onlyonce[m].append(record["message"])
                    return True
                return False
        return True

    def __call__(self, record):
        return self.filter(record)


logfilter = LogFilter(onlyonce=["SSB obs pos \[\S+ \S+ \S+\] m"])

log.remove()
log.add(sys.stderr, level="DEBUG", filter=logfilter)

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
    # "PulsarEcliptic",
    # "PulsarMJD",
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
J2000ld = time_to_longdouble(J2000)
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
