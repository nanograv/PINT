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

def topo_posvels(xyz, toarow):
    """Return a PosVel instance for the observatory at the time of the TOA

    This routine returns a PosVel instance, containing the positions
    (m) and velocities (m / UT1 s) at the time of the toa and
    referenced to the ITRF geocentric coordinates.  This routine is
    basically SOFA's pvtob() with an extra rotation from c2ixys.
    """
    toa = toarow['mjd']
    tt = toa.tt.jd1, toa.tt.jd2
    mjd = toa.mjd

    # Gets x,y coords of Celestial Intermediate Pole and CIO locator s
    X, Y, S = erfa.xys00a(*tt)

    # Get dX and dY from IERS A
    #dX = numpy.interp(mjd, iers_tab['MJD'], iers_tab['dX_2000A_B']) * u.arcsec
    #dY = numpy.interp(mjd, iers_tab['MJD'], iers_tab['dY_2000A_B']) * u.arcsec
    # Get dX and dY from IERS B
    dX = numpy.interp(mjd, iers_tab['MJD'], iers_tab['dX_2000A']) * u.arcsec
    dY = numpy.interp(mjd, iers_tab['MJD'], iers_tab['dY_2000A']) * u.arcsec

    # Get GCRS to CIRS matrix
    rc2i = erfa.c2ixys(X+dX.to(u.rad).value, Y+dY.to(u.rad).value, S)
    # Gets the TIO locator s'
    sp = erfa.sp00(*tt)
    # Get X and Y from IERS A
    #xp = numpy.interp(mjd, iers_tab['MJD'], iers_tab['PM_X_B']) * u.arcsec
    #yp = numpy.interp(mjd, iers_tab['MJD'], iers_tab['PM_Y_B']) * u.arcsec
    # Get X and Y from IERS B
    xp = numpy.interp(mjd, iers_tab['MJD'], iers_tab['PM_x']) * u.arcsec
    yp = numpy.interp(mjd, iers_tab['MJD'], iers_tab['PM_y']) * u.arcsec
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

    return utils.PosVel(pos, vel, obj=toarow['obs'], origin="EARTH")

def topo_posvels_array(TOAs):
    """
    A array version of computing topocenter
    """
    # Two double point time 2-D array for tt
    tt = TOAs.dataTable['tt']
    # Float point time  1-D numpy array for mjd
    mjd = TOAs.dataTable['utc']
    # Two double point for ut1
    ut1 = TOAs.dataTable['ut1']
    # Get x,y coords of Celestial Intermediate Pole and CIO locator s
        
    # Interp the IERS table        
    # Get dX and dY from IERS B
    dX = numpy.interp(mjd, iers_tab['MJD'], iers_tab['dX_2000A']) * u.arcsec
    dY = numpy.interp(mjd, iers_tab['MJD'], iers_tab['dY_2000A']) * u.arcsec
    # Get the TIO locator s'
    xp = numpy.interp(mjd, iers_tab['MJD'], iers_tab['PM_x']) * u.arcsec
    yp = numpy.interp(mjd, iers_tab['MJD'], iers_tab['PM_y']) * u.arcsec
        
    posvel = numpy.zeros((TOAs.NumToa,6))
    for ii in range(TOAs.NumToa):
        obsname = TOAs.dataTable['obs'][ii]
        # Get GCRS to CIRS matrix
        X, Y, S = erfa.xys00a(*(tt[ii,0],tt[ii,1]))
        rc2i = erfa.c2ixys(X+dX[ii].to(u.rad).value, Y+dY[ii].to(u.rad).value, S)    
        
        # Get the TIO locator s'
        sp = erfa.sp00(*(tt[ii,0],tt[ii,1]))
        #Get the polar motion matrix
        rpm = erfa.pom00(xp[ii].to(u.rad).value, yp[ii].to(u.rad).value, sp)
            
        # Observatory XYZ coords in meters
        xyz = getattr(TOAs,'xyz_'+obsname)
        xyzm = [a.to(u.m).value for a in xyz]
        x, y, z = erfa.trxp(rpm, xyzm)
        
        # Functions of earth Roatation Angle

        theta = erfa.era00(*(ut1[ii,0],ut1[ii,1]))
        s, c = math.sin(theta), math.cos(theta)
            
        # Position 

        pos = numpy.asarray([c*x - s*y, s*x + c*y, z])
        pos = erfa.trxp(rc2i,pos)*u.m

        # Earth rotation rate in radians per UT1 second
        OM = 1.00273781191135448 * 2 * math.pi / erfa.DAYSEC

        # Velocity
        vel = numpy.asarray([OM * (-s*x - c*y), OM * (c*x - s*y), 0.0])
        vel = erfa.trxp(rc2i, vel)*u.m/u.s 
        
        posvel[ii] = numpy.hstack((pos,vel))
    
    return posvel  






