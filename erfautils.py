import math
import astropy.units as u
import erfa

def topo_posvels(xyz, toa):
    """
    topo_posvels(xyz, toa)

    This routine returns a tuple of 2 tuples, containing
    the positions (m) and velocities (m / UT1 s) at the
    time of the toa and referenced to the ITRF geocentric
    coordinates.  This routine is basically SOFA's Pvtob().
    For higher precision, we should include the IERS corrections
    to the IAU2000 precession/nutation model which is found in
    IERS Bulletin A.
    """
    # All the times are passed as TT
    tt = toa.tt.jd1, toa.tt.jd2

    # Polar motion and TIO position
    xp, yp, ss = erfa.xys00a(*tt) # Should add the IERS Bull. A vals
    sp = erfa.sp00(*tt)
    rpm = erfa.pom00(xp, yp, sp)
    xyzm = [a.to(u.m).value for a in xyz]
    x, y, z = erfa.trxp(rpm, xyzm)

    # Functions of Earth Rotation Angle
    theta = erfa.era00(*tt)
    s, c = math.sin(theta), math.cos(theta)

    # Position
    pos = c*x - s*y, s*x + c*y, z

    # Earth rotation rate in radians per UT1 second
    OM = 1.00273781191135448 * 2 * math.pi / erfa.DAYSEC

    # Velocity
    vel = OM * (-s*x - c*y), OM * (c*x - s*y), 0.0

    return pos, vel
