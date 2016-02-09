import os
import numpy
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy import log
from pint import pintdir

class observatory(object):
    pass

def load_tempo1_clock_file(filename,site=None):
    """
    Given the specified full path to the tempo1-format clock file, 
    will return two numpy arrays containing the MJDs and the clock
    corrections.  All computations here are done as in tempo, with
    the exception of the 'F' flag (to disable interpolation), which
    is currently not implemented.

    INCLUDE statments are processed.

    If the 'site' argument is set to an appropriate one-character tempo
    site code, only values for that site will be returned, otherwise all
    values found in the file will be returned.
    """
    mjds = []
    clkcorrs = []
    for l in open(filename).readlines():
        # Ignore comment lines
        if l.startswith('#'): continue

        # Process INCLUDE
        # Assumes included file is in same dir as this one
        if l.startswith('INCLUDE'):
            clkdir = os.path.dirname(os.path.abspath(filename))
            filename1 = os.path.join(clkdir, l.split()[1])
            mjds1, clkcorrs1 = load_tempo1_clock_file(filename1,site=site)
            mjds.extend(mjds1)
            clkcorrs.extend(clkcorrs1)
            continue

        # Parse MJD
        try:
            mjd = float(l[0:9])
            if mjd<39000 or mjd>100000: mjd=None
        except (ValueError, IndexError):
            mjd = None
        # Parse two clkcorr values
        try:
            clkcorr1 = float(l[9:21])
        except (ValueError, IndexError):
            clkcorr1 = None
        try:
            clkcorr2 = float(l[21:33])
        except (ValueError, IndexError):
            clkcorr2 = None

        # Site code on clock file line must match
        try:
            csite = l[34].lower()
        except IndexError:
            csite = None
        if (site is not None) and (site.lower()!=csite): continue

        # Need MJD and at least one of the two clkcorrs
        if mjd is None: continue
        if (clkcorr1 is None) and (clkcorr2 is None): continue
        # If one of the clkcorrs is missing, it defaults to zero
        if clkcorr1 is None: clkcorr1 = 0.0
        if clkcorr2 is None: clkcorr2 = 0.0
        # This adjustment is hard-coded in tempo:
        if clkcorr1>800.0: clkcorr1 -= 818.8
        # Add the value to the list
        mjds.append(mjd)
        clkcorrs.append(clkcorr2 - clkcorr1)

    return mjds, clkcorrs
    
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
    # The following works for simple linear interpolation
    # of normal TEMPO-style clock correction files
    # Find the 1-character tempo code, this is necessary for properly
    # reading the file.
    obs = read_observatories()
    site = next((x for x in obs[obsname].aliases if len(x)==1), None)
    if site is None:
        log.error("No tempo site code for '%s', skipping clock corrections" 
                % obsname)
        return (numpy.array([0.0, 100000.0]), numpy.array([0.0, 0.0]))
    filenm = os.path.join(os.environ["TEMPO"], "clock/time.dat")
    mjds, ccorr = load_tempo1_clock_file(filenm,site=site)
    return numpy.array(mjds), numpy.array(ccorr)

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

