import os
import numpy
import astropy.units as u
from astropy.coordinates import EarthLocation
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
    
    # SUGGESTION(paulr): This docstring should specify exactly what is expected of
    # the clock correction files (i.e. the source and destination timescales.
    # Also, a routine should probably be provided to actually use the corrections, with
    # proper interpolation, instead of the current manual calculation that toa.py does
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
    if obsname == "Parkes":
        # Parkes clock correction file changes which column is used part way through
        # This skips all pre-GPS data (50844.73 = 1998 Jan 31),
        # and goes to the point where the 'GPS-PKS' column is populated
        mjds, ccorr = numpy.loadtxt(filenm, skiprows=1003,
                                    usecols=(0, 2), unpack=True)
    else:
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
                xyz = numpy.asarray([float(x) for x in vals[1:4]]) * u.m
                obs.loc = EarthLocation(*xyz)
                obs.aliases = [obs.name.upper()]+[x.upper() for x in vals[4:]]
                observatories[obs.name] = obs
    return observatories

