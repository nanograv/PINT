import re, sys, os, numpy, gzip, copy
from . import utils
from .observatory import Observatory, get_observatory
from . import erfautils
import astropy.time as time
from . import pulsar_mjd
from astropy.extern.six.moves import cPickle as pickle
import astropy.table as table
import astropy.units as u
from astropy.coordinates import EarthLocation
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
from .solar_system_ephemerides import objPosVel_wrt_SSB
from pint import ls, J2000, J2000ld
from .config import datapath
from astropy import log

toa_commands = ("DITHER", "EFAC", "EMAX", "EMAP", "EMIN", "EQUAD", "FMAX",
                "FMIN", "INCLUDE", "INFO", "JUMP", "MODE", "NOSKIP", "PHA1",
                "PHA2", "PHASE", "SEARCH", "SIGMA", "SIM", "SKIP", "TIME",
                "TRACK", "ZAWGT", "FORMAT", "END")

iers_a_file = None
iers_a = None


def get_TOAs(timfile, ephem="DE421", include_bipm=True,
             include_gps=True, planets=False, usepickle=False):
    """Convenience function to load and prepare TOAs for PINT use.

    Loads TOAs from a '.tim' file, applies clock corrections, computes
    key values (like TDB), computes the observatory position and velocity
    vectors, and pickles the file for later use (if requested).

    Includes options to specify solar system ephemeris [default DE421],
    gps clock corrections [default=True], and BIPM clock corrections
    [default=True].
    """
    updatepickle = False
    if usepickle:
        picklefile = _check_pickle(timfile)
        if picklefile:
            timfile = picklefile
        else:
            # Pickle either did not exist or is out of date
            updatepickle = True
    t = TOAs(timfile)
    if not any(['clkcorr' in f for f in t.table['flags']]):
        log.info("Applying clock corrections.")
        t.apply_clock_corrections(include_gps=include_gps,
                                  include_bipm=include_bipm)
    if 'tdb' not in t.table.colnames:
        log.info("Getting IERS params and computing TDBs.")
        t.compute_TDBs()
    if 'ssb_obs_pos' not in t.table.colnames:
        log.info("Computing observatory positions and velocities.")
        t.compute_posvels(ephem, planets)
    # Update pickle if needed:
    if usepickle and updatepickle:
        log.info("Pickling TOAs.")
        t.pickle()
    return t

def _check_pickle(toafilename, picklefilename=None):
    """Checks if pickle file for the given toafilename needs to be updated.
    Currently only file modification times are compared, note this will
    give misleading results under some circumstances.

    If picklefilename is not specified, will look for (toafilename).pickle.gz
    then (toafilename).pickle.

    If the pickle exists and is up to date, returns the pickle file name.
    Otherwise returns empty string.
    """
    if picklefilename is None:
        for ext in (".pickle.gz", ".pickle"):
            testfilename = toafilename + ext
            if os.path.isfile(testfilename):
                picklefilename = testfilename
                break
        # It it's still None, no pickles were found
        if picklefilename is None:
            return ''

    # Check if TOA is newer than pickle
    if os.path.getmtime(picklefilename) < os.path.getmtime(toafilename):
        return ''

    # TODO add more tests.  Some things to consider:
    #   1. Check file contents via md5sum (will require storing this in pickle).
    #   2. Check INCLUDEd TOA files (will require some TOA file parsing).

    # All checks passed, return name of pickle.
    return picklefilename

def get_TOAs_list(toa_list,ephem="DE421", include_bipm=True,
                  include_gps=True, planets=False):
    """Load TOAs from a list of TOA objects.

    Compute the TDB time and observatory positions and velocity
    vectors.

    Includes options to specify solar system ephemeris [default DE421],
    gps clock corrections [default=True], and BIPM clock corrections
    [default=True].
    """
    t = TOAs(toalist = toa_list)
    if not any([f.has_key('clkcorr') for f in t.table['flags']]):
        log.info("Applying clock corrections.")
        t.apply_clock_corrections(include_gps=include_gps,
                                  include_bipm=include_bipm)
    if 'tdb' not in t.table.colnames:
        log.info("Getting IERS params and computing TDBs.")
        t.compute_TDBs()
    if 'ssb_obs_pos' not in t.table.colnames:
        log.info("Computing observatory positions and velocities.")
        t.compute_posvels(ephem, planets)
    return t

def toa_format(line, fmt="Unknown"):
    """Determine the type of a TOA line.

    Identifies a TOA line as one of the following types:
    Comment, Command, Blank, Tempo2, Princeton, ITOA, Parkes, Unknown.
    """
    if re.match(r"[0-9a-z@] ", line):
        return "Princeton"
    elif line[0] == 'C' or line[0] == '#':
        return "Comment"
    elif line.startswith(toa_commands):
        return "Command"
    elif re.match(r"^\s+$", line):
        return "Blank"
    elif len(line) > 80 or fmt == "Tempo2":
        return "Tempo2"
    elif re.match(r"  ", line) and len(line) > 41 and line[41] == '.':
        return "Parkes"
    elif re.match(r"\S\S", line) and len(line) > 14 and line[14] == '.':
        # FIXME: This needs to be better
        return "ITOA"
    else:
        return "Unknown"

def get_obs(obscode):
    """Return the standard name for the given code."""
    return get_observatory(obscode).name

def parse_TOA_line(line, fmt="Unknown"):
    """Parse a one-line ASCII time-of-arrival.

    Return an MJD tuple and a dictionary of other TOA information.
    The format can be one of: Comment, Command, Blank, Tempo2,
    Princeton, ITOA, Parkes, or Unknown.
    """
    MJD = None
    fmt = toa_format(line, fmt)
    d = dict(format=fmt)
    if fmt == "Princeton":
#            Princeton format
#            ----------------
#            columns  item
#            1-1     Observatory (one-character code) '@' is barycenter
#            2-2     must be blank
#            16-24   Observing frequency (MHz)
#            25-44   TOA (decimal point must be in column 30 or column 31)
#            45-53   TOA uncertainty (microseconds)
#            69-78   DM correction (pc cm^-3)
        #fields = line.split()
        d["obs"] = get_obs(line[0].upper())
        d["freq"] = float(line[15:24])
        d["error"] = float(line[44:53])
        ii, ff = line[24:44].split('.')
        MJD = (int(ii), float("0."+ff))
        #log.info('MJD {0} {1:.12f}'.format(MJD[0],MJD[1]))
        try:
            d["ddm"] = float(line[68:78])
        except ValueError:
            d["ddm"] = 0.0
    elif fmt == "Tempo2":
        # This could use more error catching...
        fields = line.split()
        d["name"] = fields[0]
        d["freq"] = float(fields[1])
        ii, ff = fields[2].split('.')
        MJD = (int(ii), float("0."+ff))
        d["error"] = float(fields[3])
        d["obs"] = get_obs(fields[4].upper())
        # All the rest should be flags
        flags = fields[5:]
        for i in range(0, len(flags), 2):
            k, v = flags[i].lstrip('-'), flags[i+1]
            try:  # Convert what we can to floats and ints
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
    elif fmt == "Command":
        d[fmt] = line.split()
    elif fmt == "Parkes" or fmt == "ITOA":
        raise RuntimeError(
            "TOA format '%s' not implemented yet" % fmt)
    return MJD, d

def format_toa_line(toatime, toaerr, freq, obs, dm=0.0*u.pc/u.cm**3, name='unk', flags={},
    format='Princeton'):
    """
    Format TOA line for writing

    Inputs
    ------
    toatime   Time object containing TOA arrival time
    toaerr    TOA error as a Quantity with units
    freq      Frequency as a Quantity with units (NB: value of np.inf is allowed)
    obs       Observatory object

    dm        DM for the TOA as a Quantity with units (not printed if 0.0 pc/cm^3)
    name      Name to embed in TOA line (conventionally the data file name)
    format    (Princeton | Tempo2)
    flags     Any Tempo2 flags to append to the TOA line

    Bugs
    ----
    This implementation is currently incomplete in that it will not
    undo things like TIME statements and probably other things.

    Princeton format
    ----------------
    columns  item
    1-1     Observatory (one-character code) '@' is barycenter
    2-2     must be blank
    16-24   Observing frequency (MHz)
    25-44   TOA (decimal point must be in column 30 or column 31)
    45-53   TOA uncertainty (microseconds)
    69-78   DM correction (pc cm^-3)

    Tempo2 format
    -------------
    First line of file should be "FORMAT 1"
    TOA format is "file freq sat satErr siteID <flags>"

    Returns
    -------
    out : string
        Formatted TOA line
    """
    from .utils import time_to_mjd_string
    if format.upper() in ('TEMPO2','1'):
        toa_str = time_to_mjd_string(toatime,prec=16)
        #log.info(toa_str)
        # In Tempo2 format, freq=0.0 means infinite frequency
        if freq == numpy.inf*u.MHz:
            freq = 0.0*u.MHz
        flagstring = ''
        if dm != 0.0*u.pc/u.cm**3:
            flagstring += "-dm {0:%.5f}".format(dm.to(u.pc/u.cm**3).value)
        # Here I need to append any actual flags
        # Now set observatory code. Use obs.name unless overridden by tempo2_code
        try:
            obscode = obs.tempo2_code
        except:
            obscode = obs.name
        out = "%s %f %s %.3f %s %s\n" % (name,freq.to(u.MHz).value,
            toa_str,toaerr.to(u.us).value,obs.name,flagstring)
    elif format.upper() in  ('PRINCETON','TEMPO'): # TEMPO/Princeton format
        toa_str = time_to_mjd_string(toatime,prec=13)
        # In TEMPO/Princeton format, freq=0.0 means infinite frequency
        if freq == numpy.inf*u.MHz:
            freq = 0.0*u.MHz
        if len(obs.tempo_code) != 1:
            log.warn('Observatory {0} does not have 1-character tempo_code, skipping TOA!'.format(obs.name))
        if dm!=0.0*u.pc/u.cm**3:
            out = obs.tempo_code+" %13s%9.3f%20s%9.2f                %9.4f\n" % \
                (name, freq.to(u.MHz).value, toa_str, toaerr.to(u.us).value,
                dm.to(u.pc/u.cm**3).value)
        else:
            out = obs.tempo_code+" %13s%9.3f%20s%9.2f\n" % (name, freq.to(u.MHz).value,
                toa_str, toaerr.to(u.us).value)
    else:
        log.error('Unknown TOA format ({0})'.format(format))
        # Should this raise an exception here? -- paulr

    return out


class TOA(object):
    """A time of arrival (TOA) class.

        MJD will be stored in astropy.time.Time format, and can be
            passed as a double (not recommended), a string, a
            tuple of component parts (usually day and fraction of day).
        error is the TOA uncertainty in microseconds
        obs is the observatory name as defined by the Observatory class
        freq is the observatory-centric frequency in MHz
        other keyword/value pairs can be specified as needed

        A discussion of times and clock corrections in PINT is available here:
        https://github.com/nanograv/PINT/wiki/Clock-Corrections-and-Timescales-in-PINT

    Example:
        >>> a = TOA((54567, 0.876876876876876), 4.5, freq=1400.0,
        ...         obs="GBT", backend="GUPPI")
        >>> print a
        54567.876876876876876:  4.500 us error from 'GBT' at 1400.0000 MHz {'backend': 'GUPPI'}


    What happens if IERS data is not available for the date:
        >>> a = TOA((154567, 0.876876876876876), 4.5, freq=1400.0,
        ...         obs="GBT", backend="GUPPI")

        Traceback (most recent call last):
          omitted
        IndexError: (some) times are outside of range covered by IERS table.


    """
    def __init__(self, MJD, # required
                 error=0.0, obs='Barycenter', freq=float("inf"),
                 scale=None,
                 **kwargs):  # keyword args that are completely optional
        r"""
        Construct a TOA object

        Parameters
        ----------
        MJD : astropy Time, float, or tuple of floats
            The time of the TOA, which can be expressed as an astropy Time,
            a floating point MJD (64 or 128 bit precision), or a tuple
            of (MJD1,MJD2) whose sum is the full precision MJD (usually the
            integer and fractional part of the MJD)
        obs : string
            The observatory code for the TOA
        freq : float or astropy Quantity
            Frequency corresponding to the TOA.  Either a Quantity with frequency
            units, or a number for which MHz is assumed.
        scale : string
            Time scale for the TOA time.  Defaults to the timescale appropriate
            to the site, but can be overridden

        Notes
        -----
        It is VERY important that all astropy.Time() objects are created
        with precision=9. This is ensured in the code and is checked for any
        Time object passed to the TOA constructor.

        """
        site = get_observatory(obs)

        # If MJD is already a Time, just use it. Note that this will ignore
        # the 'scale' argument to the TOA() constructor!
        if isinstance(MJD,time.Time):
            t = MJD
        else:
            if numpy.isscalar(MJD):
                arg1, arg2 = MJD, None
            else:
                arg1, arg2 = MJD[0], MJD[1]
            if scale is None:
                scale = site.timescale
            # First build a time without a location
            # Note that when scale is UTC, must use pulsar_mjd format!
            if scale.lower() == 'utc':
                fmt = 'pulsar_mjd'
            else:
                fmt = 'mjd'
            t = time.Time(arg1, arg2, scale=scale,
                    format=fmt, precision=9)
        # Now assign the site location to the Time, for use in the TDB conversion
        # Time objects are immutable so you must make a new one to add the location!
        # Use the intial time to look up the observatory location
        # (needed for moving observatories)
        # The location is an EarthLocation in the ITRF (ECEF, WGS84) frame
        try:
            loc = site.earth_location_itrf(time=t)
        except:
            print("Error computing earth_location_itrf at time {0}, {1}".format(t,type(t)))
            raise
        # Then construct the full time, with observatory location set
        self.mjd = time.Time(t, location=loc, precision=9)

        if hasattr(error,'unit'):
            self.error = error
        else:
            self.error = error * u.microsecond
        self.obs = site.name
        if hasattr(freq,'unit'):
            try:
                junk = freq.to(u.MHz)
            except u.UnitConversionError:
                log.error("Frequency for TOA with incompatible unit {0}".format(freq))
            self.freq = freq
        else:
            self.freq = freq * u.MHz
        if self.freq == 0.0*u.MHz:
            self.freq = numpy.inf*u.MHz
        self.flags = kwargs


    def __str__(self):
        s = utils.time_to_mjd_string(self.mjd) + \
            ": %6.3f %s error from '%s' at %.4f %s " % \
            (self.error.value, self.error.unit, self.obs, self.freq.value,self.freq.unit)
        if len(self.flags):
            s += str(self.flags)
        return s


class TOAs(object):
    """A class of multiple TOAs, loaded from zero or more files."""

    def __init__(self, toafile=None, toalist=None):
        # First, just make an empty container
        self.toas = []
        self.commands = []
        self.filename = None
        self.planets = False

        if (toalist is not None) and (toafile is not None):
            log.error('Cannot initialize TOAs from both file and list.')

        if toafile is not None:
            # FIXME: work with file-like objects as well
            # Check for a pickle-like filename.  Alternative approach would
            # be to just try opening it as a pickle and see what happens.
            if toafile.endswith('.pickle') or toafile.endswith('pickle.gz'):
                log.info('Reading TOAs from pickle file')
                self.read_pickle_file(toafile)
            else: # Not a pickle file, process as a standard set of TOA lines
                self.read_toa_file(toafile)
                self.filename = toafile

        if toalist is not None:
            if not isinstance(toalist, (list, tuple)):
                log.error('Trying to initialize TOAs from a non-list class')
            self.toas = toalist

        if not hasattr(self, 'table'):
            mjds = self.get_mjds(high_precision=True)
            # The table is grouped by observatory
            self.table = table.Table([numpy.arange(len(mjds)), mjds, self.get_mjds(),
                                      self.get_errors(), self.get_freqs(),
                                      self.get_obss(), self.get_flags()],
                                      names=("index", "mjd", "mjd_float", "error",
                                             "freq", "obs", "flags"),
                                      meta={'filename':self.filename}).group_by("obs")

        # We don't need this now that we have a table
        del(self.toas)

    @property
    def ntoas(self):
        return len(self.table) if hasattr(self, "table") else len(self.toas)

    @property
    def observatories(self):
        return set(self.get_obss())

    @property
    def first_MJD(self):
        return self.get_mjds(high_precision=True).min()

    @property
    def last_MJD(self):
        return self.get_mjds(high_precision=True).max()
 
    def __add__(self, x):
        if type(x) in [int, float]:
            if not x:
                # Adding zero. Do nothing
                return self

    def __sub__(self, x):
        if type(x) in [int, float]:
            if not x:
                # Subtracting zero. Do nothing
                return self

    def get_freqs(self):
        """Return a numpy array of the observing frequencies in MHz for the TOAs"""
        if hasattr(self, "toas"):
            return numpy.array([t.freq.to(u.MHz).value for t in self.toas]) * u.MHz
        else:
            x = self.table['freq']
            return numpy.asarray(x) * x.unit

    def get_mjds(self, high_precision=False):
        """ With high_precision is True
            Return an array of the astropy.times (UTC) of the TOAs

            With high_precision is False
            Return an array of toas in mjd as double precision floats

            WARNING: Depending on the situation, you may get MJDs in a
            different scales (e.g. UTC, TT, or TDB) or even a mixture
            of scales if some TOAs are barycentred and some are not (a
            perfectly valid situation when fitting both Fermi and radio TOAs)
        """
        if high_precision:
            if hasattr(self, "toas"):
                return numpy.array([t.mjd for t in self.toas])
            else:
                return numpy.array([t for t in self.table['mjd']])
        else:
            if hasattr(self, "toas"):
                return numpy.array([t.mjd.mjd for t in self.toas]) * u.day
            else:
                return self.table['mjd_float']


    def get_errors(self):
        """Return a numpy array of the TOA errors in us"""
        #FIXME temporarily disable reading errors from toas
        if hasattr(self, "toas"):
            return numpy.array([t.error.to(u.us).value for t in self.toas]) * u.us
        else:
            return self.table['error']

    def get_obss(self):
        """Return a numpy array of the observatories for each TOA"""
        if hasattr(self, "toas"):
            return numpy.array([t.obs for t in self.toas])
        else:
            return self.table['obs']

    def get_flags(self):
        """Return a numpy array of the TOA flags"""
        if hasattr(self, "toas"):
            return numpy.array([t.flags for t in self.toas])
        else:
            return self.table['flags']

    def select(self, selectarray):
        """Apply a boolean selection or mask array to the TOA table."""
        if hasattr(self, "table"):
            # Allow for selection undos
            if not hasattr(self, "table_selects"):
                self.table_selects = []
            self.table_selects.append(copy.deepcopy(self.table))
            # Our TOA table must be grouped by observatory for phase calcs
            self.table = self.table[selectarray].group_by('obs')
        else:
            log.warn("TOA selection not implemented for TOA lists.")

    def unselect(self):
        """Return to previous selected version of the TOA table (stored in stack)."""
        if hasattr(self, "table_selects") and len(self.table_selects):
            self.table = self.table_selects.pop()
        else:
            log.warn("No previous TOA table found.  No changes made.")

    def pickle(self, filename=None):
        """Write the TOAs to a .pickle file with optional filename."""
        if filename is not None:
            pickle.dump(self, open(filename, "wb"))
        elif self.filename is not None:
            pickle.dump(self, gzip.open(self.filename+".pickle.gz", "wb"))
        else:
            log.warn("TOA pickle method needs a filename.")

    def get_summary(self):
        """Return a short ASCII summary of the TOAs."""
        s = "Number of TOAs:  %d\n" % self.ntoas
        s += "Number of commands:  %d\n" % len(self.commands)
        s += "Number of observatories:  %d %s\n" % (len(self.observatories),
                                                    list(self.observatories))
        s += "MJD span:  %.3f to %.3f\n" % (self.first_MJD.value,
                                            self.last_MJD.value)
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            s += "%s TOAs (%d):\n" % (key['obs'], len(grp))
            s += "  Min error:     %.3g us\n" % numpy.min(grp['error'])
            s += "  Max error:     %.3g us\n" % numpy.max(grp['error'])
            s += "  Mean error:    %.3g us\n" % numpy.mean(grp['error'])
            s += "  Median error:  %.3g us\n" % numpy.median(grp['error'])
            s += "  Error stddev:  %.3g us\n" % numpy.std(grp['error'])
        return s

    def print_summary(self):
        """Write a summary of the TOAs to stdout."""
        print(self.get_summary())

    def adjust_TOAs(self, delta):
        """Apply a time delta to TOAs

        Adjusts the time (MJD) of the TOAs by applying delta, which should
        be a numpy.time.TimeDelta instance with the same shape as self.table['mjd']

        Parameters
        ----------
        delta : astropy.time.TimeDelta
            The time difference to add to the MJD of each TOA

        """
        col = self.table['mjd']
        if type(delta) != time.TimeDelta:
            raise ValueError('Type of argument must be TimeDelta')
        if delta.shape != col.shape:
            raise ValueError('Shape of mjd column and delta must be compatible')
        for ii in range(len(col)):
            col[ii] = col[ii] + delta[ii]

        # This adjustment invalidates the derived columns in the table, so delete
        # and recompute them
        self.table['mjd_float'] = self.get_mjds(high_precision=False)
        self.compute_TDBs()
        self.compute_posvels()

    def write_TOA_file(self,filename,name='pint', format='Princeton'):
        """Dump current TOA table out as a TOA file

        Parameters
        ----------
        filename : str
            File name to write to
        format : str
            Format specifier for file ('TEMPO' or 'Princeton') or ('Tempo2' or '1')

        Bugs
        ----
        Currently does not undo any clock corrections that were applied,
        so TOA file won't match the input TOA file if any were applied.

        """
        outf = open(filename,'w')
        if format.upper() in ('TEMPO2','1'):
            outf.write('FORMAT 1\n')
        # NOTE(paulr): This really should REMOVE any(?) clock corrections
        # that have been applied!
        for toatime,toaerr,freq,obs,flags in zip(self.table['mjd'],self.table['error'].quantity,
            self.table['freq'].quantity,self.table['obs'],self.table['flags']):
            obs_obj = Observatory.get(obs)
            str = format_toa_line(toatime, toaerr, freq, obs_obj, name=name,
                flags=flags, format=format)
            outf.write(str)
        outf.close()

    def apply_clock_corrections(self, include_bipm=True,
                                include_gps=True):
        """Apply observatory clock corrections and TIME statments.

        Apply clock corrections to all the TOAs where corrections are
        available.  This routine actually changes the value of the TOA,
        although the correction is also listed as a new flag for the TOA
        called 'clkcorr' so that it can be reversed if necessary.  This
        routine also applies all 'TIME' commands and treats them exactly
        as if they were a part of the observatory clock corrections.

        Options to include GPS or BIPM clock corrections are set to True
        by default in order to give the most accurate clock corrections.

        A description of how PINT handles clock corrections and timescales is here:
        https://github.com/nanograv/PINT/wiki/Clock-Corrections-and-Timescales-in-PINT

        """
        # First make sure that we haven't already applied clock corrections
        flags = self.table['flags']
        # if any([f.has_key('clkcorr') for f in flags]):
        #     log.warn("Some TOAs have 'clkcorr' flag.  Not applying new clock corrections.")
        #     return
        # An array of all the time corrections, one for each TOA
        corr = numpy.zeros(self.ntoas) * u.s
        times = self.table['mjd']
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]['obs']
            site = get_observatory(obs, include_gps=include_gps,
                                   include_bipm=include_bipm)
            loind, hiind = self.table.groups.indices[ii:ii+2]
            # First apply any TIME statements
            for jj in range(loind, hiind):
                if 'to' in flags[jj]:
                    # TIME commands are in sec
                    # SUGGESTION(paulr): These time correction units should
                    # be applied in the parser, not here. In the table the time
                    # correction should have units.
                    corr[jj] = flags[jj]['to'] * u.s
                    times[jj] += time.TimeDelta(corr[jj])

            gcorr = site.clock_corrections(time.Time(grp['mjd']))
            for jj, cc in enumerate(gcorr):
                grp['mjd'][jj] += time.TimeDelta(cc)
            corr[loind:hiind] += gcorr
            # Now update the flags with the clock correction used
            for jj in range(loind, hiind):
                if corr[jj]:
                    flags[jj]['clkcorr'] = corr[jj]

    def compute_TDBs(self):
        """Compute and add TDB and TDB long double columns to the TOA table.

        This routine creates new columns 'tdb' and 'tdbld' in a TOA table
        for TDB times, using the Observatory locations and IERS A Earth
        rotation corrections for UT1.
        """
        log.info('Computing TDB columns.')
        if 'tdb' in self.table.colnames:
            log.info('tdb column already exists. Deleting...')
            self.table.remove_column('tdb')
        if 'tdbld' in self.table.colnames:
            log.info('tdbld column already exists. Deleting...')
            self.table.remove_column('tdbld')

        # Compute in observatory groups
        tdbs = numpy.zeros_like(self.table['mjd'])
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]['obs']
            loind, hiind = self.table.groups.indices[ii:ii+2]
            grpmjds = time.Time(grp['mjd'], location=grp['mjd'][0].location)
            grptdbs = grpmjds.tdb
            tdbs[loind:hiind] = numpy.asarray([t for t in grptdbs])

        # Now add the new columns to the table
        col_tdb = table.Column(name='tdb', data=tdbs)
        col_tdbld = table.Column(name='tdbld',
                data=[utils.time_to_longdouble(t) for t in tdbs])
        self.table.add_columns([col_tdb, col_tdbld])

    def compute_posvels(self, ephem="DE421", planets=False):
        """Compute positions and velocities of the observatories and Earth.

        Compute the positions and velocities of the observatory (wrt
        the Geocenter) and the center of the Earth (referenced to the
        SSB) for each TOA.  The JPL solar system ephemeris can be set
        using the 'ephem' parameter.  The positions and velocities are
        set with PosVel class instances which have astropy units.
        """
        # Record the planets choice for this instance
        self.planets = planets
        log.info('Compute positions and velocities of observatories and Earth (planets = {0}), using {1} ephemeris'.format(planets, ephem))
        # Remove any existing columns
        cols_to_remove = ['ssb_obs_pos', 'ssb_obs_vel', 'obs_sun_pos']
        for c in cols_to_remove:
            if c in self.table.colnames:
                log.info('Column {0} already exists. Removing...'.format(c))
                self.table.remove_column(c)
        for p in ('jupiter', 'saturn', 'venus', 'uranus'):
            name = 'obs_'+p+'_pos'
            if name in self.table.colnames:
                log.info('Column {0} already exists. Removing...'.format(name))
                self.table.remove_column(name)

        self.table.meta['ephem'] = ephem
        ssb_obs_pos = table.Column(name='ssb_obs_pos',
                                    data=numpy.zeros((self.ntoas, 3), dtype=numpy.float64),
                                    unit=u.km, meta={'origin':'SSB', 'obj':'OBS'})
        ssb_obs_vel = table.Column(name='ssb_obs_vel',
                                    data=numpy.zeros((self.ntoas, 3), dtype=numpy.float64),
                                    unit=u.km/u.s, meta={'origin':'SSB', 'obj':'OBS'})
        obs_sun_pos = table.Column(name='obs_sun_pos',
                                    data=numpy.zeros((self.ntoas, 3), dtype=numpy.float64),
                                    unit=u.km, meta={'origin':'OBS', 'obj':'SUN'})
        if planets:
            plan_poss = {}
            for p in ('jupiter', 'saturn', 'venus', 'uranus'):
                name = 'obs_'+p+'_pos'
                plan_poss[name] = table.Column(name=name,
                                    data=numpy.zeros((self.ntoas, 3), dtype=numpy.float64),
                                    unit=u.km, meta={'origin':'OBS', 'obj':p})

        # Now step through in observatory groups
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]['obs']
            loind, hiind = self.table.groups.indices[ii:ii+2]
            site = get_observatory(obs)
            tdb = time.Time(grp['tdb'])
            ssb_obs = site.posvel(tdb,ephem)
            ssb_obs_pos[loind:hiind,:] = ssb_obs.pos.T.to(u.km)
            ssb_obs_vel[loind:hiind,:] = ssb_obs.vel.T.to(u.km/u.s)
            sun_obs = objPosVel_wrt_SSB('sun',tdb,ephem) - ssb_obs
            obs_sun_pos[loind:hiind,:] = sun_obs.pos.T.to(u.km)
            if planets:
                for p in ('jupiter', 'saturn', 'venus', 'uranus'):
                    name = 'obs_'+p+'_pos'
                    dest = p
                    pv = objPosVel_wrt_SSB(dest,tdb,ephem) - ssb_obs
                    plan_poss[name][loind:hiind,:] = pv.pos.T.to(u.km)
        cols_to_add = [ssb_obs_pos, ssb_obs_vel, obs_sun_pos]
        if planets:
            cols_to_add += plan_poss.values()
        log.info('Adding columns ' + ' '.join([cc.name for cc in cols_to_add]))
        self.table.add_columns(cols_to_add)

    def read_pickle_file(self, filename):
        """Read the TOAs from the pickle file specified in filename.  Note
        the filename should include any pickle-specific extensions (ie
        ".pickle.gz" or similar), these will not be added automatically."""

        log.info("Reading pickled TOAs from '%s'..." % filename)
        if os.path.splitext(filename)[1] == '.gz':
            infile = gzip.open(filename,'rb')
        else:
            infile = open(filename,'rb')
        tmp = pickle.load(infile)
        self.filename = tmp.filename
        if hasattr(tmp, 'toas'):
            self.toas = tmp.toas
        if hasattr(tmp, 'table'):
            self.table = tmp.table.group_by("obs")
        self.commands = tmp.commands

    def read_toa_file(self, filename, process_includes=True, top=True):
        """Read the given filename and return a list of TOA objects.

        Will process INCLUDEd files unless process_includes is False.
        """
        ntoas = 0
        if top:
            self.toas = []
            self.commands = []
            self.cdict = {"EFAC": 1.0, "EQUAD": 0.0*u.us,
                          "EMIN": 0.0*u.us, "EMAX": numpy.inf*u.us,
                          "FMIN": 0.0*u.MHz, "FMAX": numpy.inf*u.MHz,
                          "INFO": None, "SKIP": False,
                          "TIME": 0.0, "PHASE": 0,
                          "PHA1": None, "PHA2": None,
                          "MODE": 1, "JUMP": [False, 0],
                          "FORMAT": "Unknown", "END": False}
        with open(filename, "r") as f:
            for l in f.readlines():
                MJD, d = parse_TOA_line(l, fmt=self.cdict["FORMAT"])
                if d["format"] == "Command":
                    cmd = d["Command"][0]
                    self.commands.append((d["Command"], ntoas))
                    if cmd == "SKIP":
                        self.cdict[cmd] = True
                        continue
                    elif cmd == "NOSKIP":
                        self.cdict["SKIP"] = False
                        continue
                    elif cmd == "END":
                        self.cdict[cmd] = True
                        break
                    elif cmd in ("TIME", "PHASE"):
                        self.cdict[cmd] += float(d["Command"][1])
                    elif cmd in ("EMIN", "EMAX","EQUAD"):
                        self.cdict[cmd] = float(d["Command"][1])*u.us
                    elif cmd in ("FMIN", "FMAX","EQUAD"):
                        self.cdict[cmd] = float(d["Command"][1])*u.MHz
                    elif cmd in ("EFAC", \
                                 "PHA1", "PHA2"):
                        self.cdict[cmd] = float(d["Command"][1])
                        if cmd in ("PHA1", "PHA2", "TIME", "PHASE"):
                            d[cmd] = d["Command"][1]
                    elif cmd == "INFO":
                        self.cdict[cmd] = d["Command"][1]
                        d[cmd] = d["Command"][1]
                    elif cmd == "FORMAT":
                        if d["Command"][1] == "1":
                            self.cdict[cmd] = "Tempo2"
                    elif cmd == "JUMP":
                        if self.cdict[cmd][0]:
                            self.cdict[cmd][0] = False
                            self.cdict[cmd][1] += 1
                        else:
                            self.cdict[cmd][0] = True
                    elif cmd == "INCLUDE" and process_includes:
                        # Save FORMAT in a tmp
                        fmt = self.cdict["FORMAT"]
                        self.cdict["FORMAT"] = "Unknown"
                        self.read_toa_file(d["Command"][1], top=False)
                        # re-set FORMAT
                        self.cdict["FORMAT"] = fmt
                    else:
                        continue
                if (self.cdict["SKIP"] or
                    d["format"] in ("Blank", "Unknown", "Comment", "Command")):
                    continue
                elif self.cdict["END"]:
                    if top:
                        # Clean up our temporaries used when reading TOAs
                        del self.cdict
                    return
                else:
                    newtoa = TOA(MJD, **d)
                    if ((self.cdict["EMIN"] > newtoa.error) or
                        (self.cdict["EMAX"] < newtoa.error) or
                        (self.cdict["FMIN"] > newtoa.freq) or
                        (self.cdict["FMAX"] < newtoa.freq)):
                        continue
                    else:
                        newtoa.error *= self.cdict["EFAC"]
                        newtoa.error = numpy.hypot(newtoa.error,
                                                   self.cdict["EQUAD"])
                        if self.cdict["INFO"]:
                            newtoa.flags["info"] = self.cdict["INFO"]
                        if self.cdict["JUMP"][0]:
                            newtoa.flags["jump"] = self.cdict["JUMP"][1]
                        if self.cdict["PHASE"] != 0:
                            newtoa.flags["phase"] = self.cdict["PHASE"]
                        if self.cdict["TIME"] != 0.0:
                            newtoa.flags["to"] = self.cdict["TIME"]
                        self.toas.append(newtoa)
                        ntoas += 1
            if top:
                # Clean up our temporaries used when reading TOAs
                del self.cdict
