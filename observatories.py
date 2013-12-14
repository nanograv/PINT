import os
import numpy
import spice
import astropy.units as u
from astropy.coordinates.angles import Angle

class observatory:
    pass

def ITRF_to_GEO_WGS84(xyz):
    """
    ITRF_to_GEO_WGS84(xyz):
        Convert ITRF x,y,z rectangular coords (m) to WGS-84 referenced
        lat, lon, height (using astropy units).
    """
    # see http://en.wikipedia.org/wiki/World_Geodetic_System for constants
    Re_wgs84, f_wgs84 = 6378137.0, 1.0/298.257223563
    lat, lon, hgt = spice.recgeo(xyz, Re_wgs84, f_wgs84)
    return [Angle(lat, unit=u.rad), Angle(lon, unit=u.rad), hgt * u.m]

def read_observatories():
    observatories = {}
    filenm = os.path.join(os.getenv("PINT"), "datafiles/observatories.txt")
    with open(filenm) as f:
        for line in f.readlines():
            if line[0]!="#":
                vals = line.split()
                obs = observatory()
                obs.name = vals[0]
                xyz_vals = (float(vals[1]), float(vals[2]), float(vals[3]))
                obs.XYZ = [a * u.m for a in xyz_vals]
                obs.geo = ITRF_to_GEO_WGS84(xyz_vals)
                obs.aliases = [obs.name.upper()]+vals[4:]
                observatories[obs.name] = obs
    return observatories

