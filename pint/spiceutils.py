import spice
import sys
import numpy as np
import astropy.units as u
from astropy.coordinates.angles import Angle
from astropy.coordinates import Longitude, Latitude
from .utils import PosVel

def loadKernel(filenames):
    """
    loadKernel(filenames)
    
    Read SPICE kernels in filenames (including full paths)
    """
    if type(filenames) is str:
        spice.furnsh(filenames)
    else:
        for name in filenames:
            spice.furnsh(name)

def objPosVel(obj1, obj2, et):
    """
    objPosVel(obj1, obj2, et)

    Returns position/velocity vectors between obj1 and obj2 as a 
    PosVel object at the given time.

    et is in spice format (TDB seconds past J2000 epoch)
    """
    # TODO: 
    #  - maybe this should be a PosVel __init__ method instead?
    #  - accept an astropy time rather than et as input?
    pv, lt = spice.spkezr(obj1, float(et), "J2000", "NONE", obj2)
    return PosVel(pv[:3]*u.km, pv[3:]*u.km/u.s, obj=obj1, origin=obj2)


def objPosVel2SSB(objname, et):
    """
    objPosVel2SSB(objname, et)
    
    Returns a solar system object position and velocity in J2000 SSB
    coordinates.  Requires SPK and LPSK kernels in J2000 SSB coordinates.
    """
    return spice.spkezr(objname.upper(), et, "J2000", "NONE", "SSB")


def getobsJ2000(posITRF, et):
    """
    getobsJ2000(posITRF, et)

    Returns observatory rectangular coordinates in J2000 Earth
    centered coordinates.  Requires PCK kernels.  posITRF is a double
    precision vector of [x,y,z] in ITRF coordinate in km.
    """
    state = np.array(posITRF+[0,0,0])
    # Transformation matrix from ITRF93 to J2000
    # CALL SXFORM(COORDINATE FROM, COORDINATE TO, ET)
    xform = np.matrix(spice.sxform("ITRF93", "J2000", et))
    # Coordinate transformation.  First three elements are position
    # [x,y,z] in J2000 earth centered, second three elements are
    # velocity [dx/dt,dy/dt,dz/dt] in J2000 earth centered
    return np.dot(xform, state)


def ITRF2GEO(posITRF):
    '''
    ITRF2GEO(posITRF)
    
    Converts from earth rectangular coordinate to Geodetic coordinate,
    Input will be the rectangular three coordinates [x,y,z].  Kernel
    file PCK is required.
    '''
    dim, value = spice.bodvcd(399, "RADII", 3)
    # Reuturns Earh radii [larger equatorial radius, smaller
    # equatorial radius, polar radius] dim is the dimension of
    # returned values Value is the returned values
    rEquatr = value[0]; 
    rPolar = value[2];
    # Calculate the flattening factor for earth 
    f = (rEquatr - rPolar) / rEquatr
    # Calculate the geodetic coordinate on earth. lon,lat are in
    # radius. alt is the same unit with in put posITRF
    lon, lat, alt = spice.recgeo(posITRF, rEquatr, f)
    # Return longitude and latitude in degree
    lonDeg = spice.convrt(lon, "RADIANS", "DEGREES")
    latDeg = spice.convrt(lat, "RADIANS", "DEGREES")
    return lon, lonDeg, lat, latDeg, alt


def ITRF_to_GEO_WGS84(x, y, z):
    """
    ITRF_to_GEO_WGS84(x, y, z)

    Convert ITRF x, y, z rectangular coords (m) to WGS-84 referenced
    lon, lat, height (using astropy units).
    """
    # see http://en.wikipedia.org/wiki/World_Geodetic_System for constants
    Re_wgs84, f_wgs84 = 6378137.0, 1.0/298.257223563
    lon, lat, hgt = spice.recgeo((x.to(u.m).value,
                                  y.to(u.m).value,
                                  z.to(u.m).value), Re_wgs84, f_wgs84)
    return Longitude(lon, 'radian', wrap_angle=180.0*u.degree), \
           Latitude(lat, 'radian'), hgt * u.m

