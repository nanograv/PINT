import os
import numpy
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy import log
from pint import pintdir

class observatory(object):
    pass

def read_clock_file(filen):
    file = open(filen)
    mjds = []
    ccorr = []
    for l in file.readlines():
        try:
            m = float(l[0:9])
            c = float(l[24:33])
            mjds.append(m)
            ccorr.append(c)
        except:
            pass

    return numpy.asarray(mjds), numpy.asarray(ccorr)


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

    mjds, ccorr = read_clock_file(filenm)

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

