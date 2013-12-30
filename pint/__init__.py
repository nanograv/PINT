# Top-level PINT __init__.py
"""
PINT Is Not TEMPO3!
"""

# Define a few important constants
import astropy.units as u
import astropy.constants as c

# light-second unit
ls = u.def_unit('ls', c.c * 1.0 * u.s)

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

# For use in models.solar_system_shapiro
ss_mass_sec = {"sun": Tsun.value,
               "mercury": Tmercury.value,
               "venus": Tvenus.value,
               "earth": Tearth.value,
               "mars": Tmars.value,
               "jupiter": Tjupiter.value,
               "saturn": Tsaturn.value,
               "uranus": Turanus.value,
               "neptune": Tneptune.value}
