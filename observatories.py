import os
import numpy
import spice
import astropy.units as u
from astropy.coordinates.angles import Angle

class observatory:
    pass

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
                obs.aliases = [obs.name.upper()]+[x.upper() for x in vals[4:]]
                observatories[obs.name] = obs
    return observatories

