from __future__ import absolute_import, print_function, division

import numpy as np
from astropy import log
try:
    import astropy.erfa as erfa
except ImportError:
    import astropy._erfa as erfa
import astropy.table as table
from astropy.time import Time
import astropy.units as u

from pint.observatory import Observatory
from pint import toa, utils, erfautils
from pinttestdata import testdir, datadir

# This is the old erfautils module, that uses IERS data files directly
# The implementation was moved here so we can remove it and use astropy
# functions, checking them against this

SECS_PER_DAY = erfa.DAYSEC

from astropy.utils.iers import IERS_A, IERS_A_URL, IERS_B, IERS_B_URL, IERS, IERS_Auto
from astropy.utils.data import download_file
# iers_a_file = download_file(IERS_A_URL, cache=True)
#iers_b_file = download_file(IERS_B_URL, cache=True)
# iers_a = IERS_A.open(iers_a_file)
#iers_b = IERS_B.open(iers_b_file)
#IERS.iers_table = iers_b
#iers_tab = IERS.iers_table
iers_tab = IERS_Auto.open()
iers_b_tab = IERS_B.open()

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

def gcrs_posvel_from_itrf(loc, toas, obsname='obs'):
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
    # If the input is a single TOA (i.e. a row from the table),
    # then put it into a list
    if type(toas) == table.row.Row:
        ttoas = Time([toas['mjd']])
    elif type(toas) == table.table.Table:
        ttoas = toas['mjd']
    elif isinstance(toas,Time):
        if toas.isscalar:
            ttoas = Time([toas])
        else:
            ttoas = toas
    else:
        if np.isscalar(toas):
            ttoas = Time([toas],format="mjd")
        else:
            ttoas = toas
    N = len(ttoas)

    # Get various times from the TOAs as arrays
    tts = np.asarray([(t.jd1, t.jd2) for t in ttoas.tt]).T
    ut1s = np.asarray([(t.jd1, t.jd2) for t in ttoas.ut1]).T
    mjds = np.asarray(ttoas.mjd)

    # Get x, y coords of Celestial Intermediate Pole and CIO locator s
    X, Y, S = erfa.xys00a(*tts)

    # Get dX and dY from IERS A in arcsec and convert to radians
    #dX = np.interp(mjds, iers_tab['MJD'], iers_tab['dX_2000A_B']) * asec2rad
    #dY = np.interp(mjds, iers_tab['MJD'], iers_tab['dY_2000A_B']) * asec2rad
    # Get dX and dY from IERS B in arcsec and convert to radians
    dX = np.interp(mjds, iers_tab['MJD'], iers_tab['dX_2000A_B'],
            left=np.nan, right=np.nan) * asec2rad
    dY = np.interp(mjds, iers_tab['MJD'], iers_tab['dY_2000A_B'],
            left=np.nan, right=np.nan) * asec2rad
    dX_B = np.interp(mjds, iers_b_tab['MJD'], iers_b_tab['dX_2000A'],
            left=np.nan, right=np.nan) * asec2rad
    dY_B = np.interp(mjds, iers_b_tab['MJD'], iers_b_tab['dY_2000A'],
            left=np.nan, right=np.nan) * asec2rad
    assert (dX, dY) == (dX_B,dY_B)

    # Get GCRS to CIRS matrices
    rc2i = erfa.c2ixys(X+dX, Y+dY, S)

    # Gets the TIO locator s'
    sp = erfa.sp00(*tts)

    # Get X and Y from IERS A in arcsec and convert to radians
    #xp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_X_B']) * asec2rad
    #yp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_Y_B']) * asec2rad
    # Get X and Y from IERS B in arcsec and convert to radians
    xp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_X_B'],
            left=np.nan, right=np.nan) * asec2rad
    yp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_Y_B'],
            left=np.nan, right=np.nan) * asec2rad

    # Get the polar motion matrices
    rpm = erfa.pom00(xp, yp, sp)

    # Observatory geocentric coords in m
    xyzm = np.array([a.to(u.m).value for a in loc.geocentric])
    x, y, z = np.dot(xyzm, rpm).T

    # Functions of Earth Rotation Angle
    theta = erfa.era00(*ut1s)
    s, c = np.sin(theta), np.cos(theta)
    sx, cx = s * x, c * x
    sy, cy = s * y, c * y

    # Initial positions and velocities
    iposs = np.asarray([cx - sy, sx + cy, z]).T
    ivels = np.asarray([OM * (-sx - cy), OM * (cx - sy), \
                        np.zeros_like(x)]).T
    # There is probably a way to do this with np.einsum or something...
    # and here it is .
    poss = np.empty((N, 3), dtype=np.float64)
    vels = np.empty((N, 3), dtype=np.float64)
    poss = np.einsum('ij,ijk->ik', iposs, rc2i)
    vels = np.einsum('ij,ijk->ik', ivels, rc2i)
    return utils.PosVel(poss.T * u.m, vels.T * u.m / u.s, obj=obsname, origin="earth")


def test_erfautils_compare_to_direct_implementation():
    o = "Arecibo"
    loc = Observatory.get(o).earth_location_itrf()
    # is this for loop really needed?
    for mjd in [56000.,56500.,57000.]:
        t = Time(mjd, scale="tdb", format="mjd")
        local_posvel = gcrs_posvel_from_itrf(
            loc, t, obsname=o)
        posvel = erfautils.gcrs_posvel_from_itrf(
            loc, t, obsname=o)
        dopv = local_posvel - posvel
        dpos = np.sqrt(np.dot(dopv.pos.to(u.m)[:,0], dopv.pos.to(u.m)[:,0]))
        dvel = np.sqrt(np.dot(dopv.vel.to(u.mm/u.s)[:,0], dopv.vel.to(u.mm/u.s)[:,0]))
        assert dpos<2, "position difference in meters"
        assert dvel<0.02, "velocity difference in mm/s"

def test_iers_discrepancies():
    iers_auto = IERS_Auto.open()
    iers_b = IERS_B.open()
    for mjd in [56000,56500,57000]:
        t = Time(mjd, scale="tdb", format="mjd")
        assert iers_b.pm_xy(t) == iers_auto.pm_xy(t)
