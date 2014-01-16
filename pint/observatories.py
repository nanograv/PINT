import os
import numpy
from . import spiceutils
import astropy.units as u
from astropy import log
from pint import pintdir

class observatory(object):
    pass

def get_clock_corr_vals(obsname, **kwargs):
    """
    get_clock_corr_vals(obsname, **kwargs)

    Return a tuple of numpy arrays of MJDs and clock
    corrections (in us) which can be used to interpolate
    a more exact clock correction for a TOA.  the kwargs are
    used if there are other things which determine the values
    (for example, backend specific corrections)
    """
    fileparts = {"GBT": "gbt",
                 "Arecibo": "ao",
                 "JVLA": "vla",
                 "Parkes": "pks",
                 "Nancay": "nancay",
                 "Effelsberg": "bonn",
                 "WSRT": "wsrt"}
    if obsname in fileparts.keys():
        filenm = os.path.join(os.environ["TEMPO"],
                              "clock/time_%s.dat" % \
                              fileparts[obsname])
    else:
        log.error("No clock correction valus for %s" % obsname)
        return (numpy.array([0.0, 100000.0]), numpy.array([0.0, 0.0]))
    # The following works for simple linear interpolation
    # of normal TEMPO-style clock correction files
    mjds, ccorr = numpy.loadtxt(filenm, skiprows=2,
                                usecols=(0, 2), unpack=True)
    return mjds, ccorr

def read_observatories():
    """Load observatory data files and return them.

    Return a dictionary of instances of the observatory class that are
    stored in the $PINT/datafiles/observatories.txt file.
    """
    observatories = {}
    filenm = os.path.join(pintdir, "datafiles/observatories.txt")
    with open(filenm) as f:
        for line in f.readlines():
            if line[0] != "#":
                vals = line.split()
                obs = observatory()
                obs.name = vals[0]
                xyz_vals = (float(vals[1]), float(vals[2]), float(vals[3]))
                obs.xyz = [a * u.m for a in xyz_vals]
                obs.geo = spiceutils.ITRF_to_GEO_WGS84(*obs.xyz)
                obs.aliases = [obs.name.upper()]+[x.upper() for x in vals[4:]]
                observatories[obs.name] = obs
    return observatories

