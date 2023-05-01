"""Observatory position and velocity calculation."""

import erfa

import numpy as np
from astropy import table
from loguru import logger as log

from pint.pulsar_mjd import Time
from pint.utils import PosVel


__all__ = ["gcrs_posvel_from_itrf"]

SECS_PER_DAY = erfa.DAYSEC
# Earth rotation rate in radians per UT1 second
#
# This is from Capitaine, Guinot, McCarthy, 2000 and is
# in IAU Resolution B1.8 on the Earth Rotation Angle (ERA)
# and the relation of it to UT1.  The number 1.00273781191135448
# below is a defining constant.  See here:
# http://iau-c31.nict.go.jp/pdf/2009_IAUGA_JD6/JD06_capitaine_wallace.pdf
OM = 1.00273781191135448 * 2.0 * np.pi / SECS_PER_DAY
# arcsec to radians
asec2rad = 4.84813681109536e-06


def gcrs_posvel_from_itrf(loc, toas, obsname="obs"):
    """Return a list of PosVel instances for the observatory at the TOA times.

    Observatory location should be given in the loc argument as an astropy
    EarthLocation object. This location will be in the ITRF frame (i.e.
    co-rotating with the Earth).

    The optional obsname argument will be used as label in the returned
    PosVel instance.

    This routine returns a list of PosVel instances, containing the
    positions (m) and velocities (m / s) at the times of the toas and
    referenced to the Earth-centered Inertial (ECI, aka GCRS) coordinates.
    This routine is basically SOFA's pvtob() [Position and velocity of
    a terrestrial observing station] with an extra rotation from c2ixys()
    [Form the celestial to intermediate-frame-of-date matrix given the CIP
    X,Y and the CIO locator s].

    This version uses astropy's internal routines, which use IERS A data
    rather than the final IERS B values. These do differ, and yield results
    that are different by ~20 m.
    """
    unpack = False
    # If the input is a single TOA (i.e. a row from the table),
    # then put it into a list
    if type(toas) == table.row.Row:
        ttoas = Time([toas["mjd"]])
        unpack = True
    elif type(toas) == table.table.Table:
        ttoas = toas["mjd"]
    elif isinstance(toas, Time):
        if toas.isscalar:
            ttoas = Time([toas])
            unpack = True
        else:
            ttoas = toas
    elif np.isscalar(toas):
        ttoas = Time([toas], format="mjd")
        unpack = True
    else:
        ttoas = toas
    t = ttoas

    pos, vel = loc.get_gcrs_posvel(t)
    r = PosVel(pos.xyz, vel.xyz, obj=obsname, origin="earth")
    return r[0] if unpack else r
