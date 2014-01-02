import math, utils, numpy
import astropy.units as u
import erfa

from astropy.utils.iers import IERS_A, IERS_A_URL, IERS_B, IERS_B_URL, IERS
from astropy.utils.data import download_file
# iers_a_file = download_file(IERS_A_URL, cache=True)
iers_b_file = download_file(IERS_B_URL, cache=True)
# iers_a = IERS_A.open(iers_a_file)
iers_b = IERS_B.open(iers_b_file)
IERS.iers_table = iers_b
iers_tab = IERS.iers_table

def topo_posvels(xyz, toa):
    """
    topo_posvels(xyz, toa)

    This routine returns a PosVel instance , containing the positions
    (m) and velocities (m / UT1 s) at the time of the toa and
    referenced to the ITRF geocentric coordinates.  This routine is
    basically SOFA's pvtob() with an extra rotation from c2ixys.
    """
    # All the times are passed as TT
    tt = toa.tt.jd1, toa.tt.jd2

    # Gets x,y coords of Celestial Intermediate Pole and CIO locator s
    X, Y, S = erfa.xys00a(*tt)
    # Get dX and dY from IERS A
    #dX = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['dX_2000A_B']) * u.arcsec
    #dY = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['dY_2000A_B']) * u.arcsec
    # Get dX and dY from IERS B
    dX = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['dX_2000A']) * u.arcsec
    dY = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['dY_2000A']) * u.arcsec
    # Get GCRS to CIRS matrix
    rc2i = erfa.c2ixys(X+dX.to(u.rad).value, Y+dY.to(u.rad).value, S)

    # Gets the TIO locator s'
    sp = erfa.sp00(*tt)
    # Get X and Y from IERS A
    #xp = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['PM_X_B']) * u.arcsec
    #yp = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['PM_Y_B']) * u.arcsec
    # Get X and Y from IERS B
    xp = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['PM_x']) * u.arcsec
    yp = numpy.interp(toa.utc.mjd, iers_tab['MJD'], iers_tab['PM_y']) * u.arcsec
    # Get the polar motion matrix
    rpm = erfa.pom00(xp.to(u.rad).value, yp.to(u.rad).value, sp)

    # Observatory XYZ coords in meters
    xyzm = [a.to(u.m).value for a in xyz]
    x, y, z = erfa.trxp(rpm, xyzm)

    # Functions of Earth Rotation Angle
    ut1 = toa.ut1.jd1, toa.ut1.jd2
    theta = erfa.era00(*ut1)
    s, c = math.sin(theta), math.cos(theta)

    # Position
    pos = numpy.asarray([c*x - s*y, s*x + c*y, z])
    pos = erfa.trxp(rc2i, pos) * u.m

    # Earth rotation rate in radians per UT1 second
    OM = 1.00273781191135448 * 2 * math.pi / erfa.DAYSEC

    # Velocity
    vel = numpy.asarray([OM * (-s*x - c*y), OM * (c*x - s*y), 0.0])
    vel = erfa.trxp(rc2i, vel) * u.m / u.s

    return utils.PosVel(pos, vel)
