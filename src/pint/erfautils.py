"""Observatory position and velocity calculation."""
from __future__ import absolute_import, division, print_function

import astropy._erfa as erfa
import astropy.units as u
from astropy import log
import numpy as np
from astropy import table
from astropy.utils.data import clear_download_cache, download_file, is_url_in_cache
from astropy.utils.iers import IERS_B, IERS_B_URL, IERS_Auto
import astropy.version
import astropy

if astropy.version.major < 4:
    log.warning(
        "Using astropy version {}. To get most recent IERS data, upgrade to astropy >= 4.0".format(
            astropy.__version__
        )
    )
else:
    from astropy.utils.iers import earth_orientation_table

from pint.pulsar_mjd import Time
from pint.utils import PosVel

__all__ = ["get_iers_up_to_date", "gcrs_posvel_from_itrf", "old_gcrs_posvel_from_itrf"]


def get_iers_up_to_date(mjd=Time.now().mjd - 45.0):
    """
    Update the IERS B table to include MJD (defaults to 45 days ago) and open IERS_Auto

    """

    # First clear the IERS_Auto table
    IERS_Auto.iers_table = None

    if mjd > Time.now().mjd:
        raise ValueError("IERS B data requested for future MJD {}".format(mjd))
    might_be_old = is_url_in_cache(IERS_B_URL)
    iers_b = IERS_B.open(download_file(IERS_B_URL, cache=True))
    if might_be_old and iers_b[-1]["MJD"].to_value(u.d) < mjd:
        # Try wiping the download and re-downloading
        log.info("IERS B Table appears to be old. Attempting to re-download.")
        clear_download_cache(IERS_B_URL)
        iers_b = IERS_B.open(download_file(IERS_B_URL, cache=True))
    if iers_b[-1]["MJD"].to_value(u.d) < mjd:
        log.warning("IERS B data not yet available for MJD {}".format(mjd))

    # Now open IERS_Auto with no argument, so it should use the IERS_B that we just made sure was up to date
    iers_auto = IERS_Auto.open()

    if astropy.version.major >= 4:
        # Tell astropy to use this table for all future transformations
        earth_orientation_table.set(iers_auto)


# This version is outdated since astropy now includes IERS_Auto (see improved version above)
def get_iers_b_up_to_date(mjd):
    """Update the IERS B table to include MJD if necessary

    """
    if Time.now().mjd <= mjd:
        raise ValueError("IERS B data requested for future MJD {}".format(mjd))
    might_be_old = is_url_in_cache(IERS_B_URL)
    iers_b = IERS_B.open(download_file(IERS_B_URL, cache=True))
    if might_be_old and iers_b[-1]["MJD"].to_value(u.d) < mjd:
        # Try wiping the download and re-downloading
        clear_download_cache(IERS_B_URL)
        iers_b = IERS_B.open(download_file(IERS_B_URL, cache=True))
    if iers_b[-1]["MJD"].to_value(u.d) < mjd:
        raise ValueError("IERS B data not yet available for MJD {}".format(mjd))
    return iers_b


# On import, make sure the IERS table is updated.
log.info("Running get_iers_up_to_date() to update IERS B table")
get_iers_up_to_date()

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


def old_gcrs_posvel_from_itrf(loc, toas, obsname="obs"):
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
    else:
        if np.isscalar(toas):
            ttoas = Time([toas], format="mjd")
            unpack = True
        else:
            ttoas = toas
    N = len(ttoas)
    if len(ttoas.shape) != 1:
        raise ValueError(
            "At most one-dimensional array of times possible, "
            "shape was {}".format(ttoas.shape)
        )

    # Get various times from the TOAs as arrays
    tts = np.asarray([(t.jd1, t.jd2) for t in ttoas.tt]).T
    ut1s = np.asarray([(t.jd1, t.jd2) for t in ttoas.ut1]).T
    mjds = np.asarray(ttoas.mjd)

    iers_b = get_iers_b_up_to_date(mjds.max())

    # Get x, y coords of Celestial Intermediate Pole and CIO locator s
    X, Y, S = erfa.xys00a(*tts)

    # Get dX and dY from IERS A in arcsec and convert to radians
    # dX = np.interp(mjds, iers_tab['MJD'], iers_tab['dX_2000A_B']) * asec2rad
    # dY = np.interp(mjds, iers_tab['MJD'], iers_tab['dY_2000A_B']) * asec2rad
    # Get dX and dY from IERS B in arcsec and convert to radians
    dX = np.interp(
        mjds, iers_b["MJD"].to_value(u.d), iers_b["dX_2000A"].to_value(u.rad)
    )
    dY = np.interp(
        mjds, iers_b["MJD"].to_value(u.d), iers_b["dY_2000A"].to_value(u.rad)
    )

    # Get GCRS to CIRS matrices
    rc2i = erfa.c2ixys(X + dX, Y + dY, S)

    # Gets the TIO locator s'
    sp = erfa.sp00(*tts)

    # Get X and Y from IERS A in arcsec and convert to radians
    # xp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_X_B']) * asec2rad
    # yp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_Y_B']) * asec2rad
    # Get X and Y from IERS B in arcsec and convert to radians
    xp = np.interp(mjds, iers_b["MJD"].to_value(u.d), iers_b["PM_x"].to_value(u.rad))
    yp = np.interp(mjds, iers_b["MJD"].to_value(u.d), iers_b["PM_y"].to_value(u.rad))

    # Get the polar motion matrices
    rpm = erfa.pom00(xp, yp, sp)

    # Observatory geocentric coords in m
    xyzm = np.array([a.to_value(u.m) for a in loc.geocentric])
    x, y, z = np.dot(xyzm, rpm).T

    # Functions of Earth Rotation Angle
    theta = erfa.era00(*ut1s)
    s, c = np.sin(theta), np.cos(theta)
    sx, cx = s * x, c * x
    sy, cy = s * y, c * y

    # Initial positions and velocities
    iposs = np.asarray([cx - sy, sx + cy, z]).T
    ivels = np.asarray([OM * (-sx - cy), OM * (cx - sy), np.zeros_like(x)]).T
    # There is probably a way to do this with np.einsum or something...
    # and here it is .
    poss = np.empty((N, 3), dtype=np.float64)
    vels = np.empty((N, 3), dtype=np.float64)
    poss = np.einsum("ij,ijk->ik", iposs, rc2i)
    vels = np.einsum("ij,ijk->ik", ivels, rc2i)
    r = PosVel(poss.T * u.m, vels.T * u.m / u.s, obj=obsname, origin="earth")
    if unpack:
        return r[0]
    else:
        return r


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
    else:
        if np.isscalar(toas):
            ttoas = Time([toas], format="mjd")
            unpack = True
        else:
            ttoas = toas
    t = ttoas

    pos, vel = loc.get_gcrs_posvel(t)
    r = PosVel(pos.xyz, vel.xyz, obj=obsname, origin="earth")
    if unpack:
        return r[0]
    else:
        return r
