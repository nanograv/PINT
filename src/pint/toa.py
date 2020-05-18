"""Tools for working with pulse time-of-arrival (TOA) data.

In particular, single TOAs are represented by :class:`pint.toa.TOA` objects, and if you
want to manage a collection of these we recommend you use a :class:`pint.toa.TOAs` object
as this makes certain operations much more convenient.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import gzip
import os
import re
from collections import OrderedDict

import astropy.table as table
import astropy.time as time
import astropy.units as u
import numpy as np
from astropy import log
from astropy.coordinates import EarthLocation
from six.moves import cPickle as pickle

from pint.observatory import Observatory, get_observatory
from pint.observatory.special_locations import SpacecraftObs
from pint.observatory.topo_obs import TopoObs
from pint.pulsar_mjd import Time
from pint.solar_system_ephemerides import objPosVel_wrt_SSB

__all__ = [
    "get_TOAs",
    "get_TOAs_list",
    "format_toa_line",
    "make_fake_toas",
    "TOA",
    "TOAs",
]

toa_commands = (
    "DITHER",
    "EFAC",
    "EMAX",
    "EMAP",
    "EMIN",
    "EQUAD",
    "FMAX",
    "FMIN",
    "INCLUDE",
    "INFO",
    "JUMP",
    "MODE",
    "NOSKIP",
    "PHA1",
    "PHA2",
    "PHASE",
    "SEARCH",
    "SIGMA",
    "SIM",
    "SKIP",
    "TIME",
    "TRACK",
    "ZAWGT",
    "FORMAT",
    "END",
)

iers_a_file = None
iers_a = None
JD_MJD = 2400000.5


def get_TOAs(
    timfile,
    ephem=None,
    include_bipm=True,
    bipm_version="BIPM2015",
    include_gps=True,
    planets=False,
    usepickle=False,
    tdb_method="default",
):
    """Convenience function to load and prepare TOAs for PINT use.

    Loads TOAs from a '.tim' file, applies clock corrections, computes
    key values (like TDB), computes the observatory position and velocity
    vectors, and pickles the file for later use (if requested).

    Includes options to specify solar system ephemeris
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
    if not any(["clkcorr" in f for f in t.table["flags"]]):
        t.apply_clock_corrections(
            include_gps=include_gps,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
        )
    if "tdb" not in t.table.colnames:
        t.compute_TDBs(method=tdb_method, ephem=ephem)
    if "ssb_obs_pos" not in t.table.colnames:
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
            return ""

    # Check if TOA is newer than pickle
    if os.path.getmtime(picklefilename) < os.path.getmtime(toafilename):
        return ""

    # TODO add more tests.  Some things to consider:
    #   1. Check file contents via md5sum (will require storing this in pickle).
    #   2. Check INCLUDEd TOA files (will require some TOA file parsing).

    # All checks passed, return name of pickle.
    return picklefilename


def get_TOAs_list(
    toa_list,
    ephem=None,
    include_bipm=True,
    bipm_version="BIPM2015",
    include_gps=True,
    planets=False,
    tdb_method="default",
):
    """Load TOAs from a list of TOA objects.

    Compute the TDB time and observatory positions and velocity
    vectors.

    Includes options to specify solar system ephemeris [default DE421],
    gps clock corrections [default=True], and BIPM clock corrections
    [default=True].
    """
    t = TOAs(toalist=toa_list)
    if not any(["clkcorr" in f for f in t.table["flags"]]):
        t.apply_clock_corrections(
            include_gps=include_gps,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
        )
    if "tdb" not in t.table.colnames:
        t.compute_TDBs(method=tdb_method, ephem=ephem)
    if "ssb_obs_pos" not in t.table.colnames:
        t.compute_posvels(ephem, planets)
    return t


def _toa_format(line, fmt="Unknown"):
    """Determine the type of a TOA line.

    Identifies a TOA line as one of the following types:
    Comment, Command, Blank, Tempo2, Princeton, ITOA, Parkes, Unknown.
    """
    if re.match(r"[0-9a-z@] ", line):
        return "Princeton"
    elif line.startswith("C ") or line.startswith("c ") or line[0] == "#":
        return "Comment"
    elif line.upper().startswith(toa_commands):
        return "Command"
    elif re.match(r"^\s+$", line):
        return "Blank"
    elif re.match(r"  ", line) and len(line) > 41 and line[41] == ".":
        return "Parkes"
    elif len(line) > 80 or fmt == "Tempo2":
        return "Tempo2"
    elif re.match(r"\S\S", line) and len(line) > 14 and line[14] == ".":
        # FIXME: This needs to be better
        return "ITOA"
    else:
        return "Unknown"


def _parse_TOA_line(line, fmt="Unknown"):
    """Parse a one-line ASCII time-of-arrival.

    Return an MJD tuple and a dictionary of other TOA information.
    The format can be one of: Comment, Command, Blank, Tempo2,
    Princeton, ITOA, Parkes, or Unknown.

    """
    MJD = None
    fmt = _toa_format(line, fmt)
    d = dict(format=fmt)
    if fmt == "Princeton":
        # Princeton format
        # ----------------
        # columns  item
        # 1-1     Observatory (one-character code) '@' is barycenter
        # 2-2     must be blank
        # 16-24   Observing frequency (MHz)
        # 25-44   TOA (decimal point must be in column 30 or column 31)
        # 45-53   TOA uncertainty (microseconds)
        # 69-78   DM correction (pc cm^-3)
        d["obs"] = get_observatory(line[0].upper()).name
        d["freq"] = float(line[15:24])
        d["error"] = float(line[44:53])
        ii, ff = line[24:44].split(".")
        MJD = (int(ii), float("0." + ff))
        try:
            d["ddm"] = float(line[68:78])
        except ValueError:
            d["ddm"] = 0.0
    elif fmt == "Tempo2":
        # This could use more error catching...
        fields = line.split()
        d["name"] = fields[0]
        d["freq"] = float(fields[1])
        ii, ff = fields[2].split(".")
        MJD = (int(ii), float("0." + ff))
        d["error"] = float(fields[3])
        d["obs"] = get_observatory(fields[4].upper()).name
        # All the rest should be flags
        flags = fields[5:]
        for i in range(0, len(flags), 2):
            k, v = flags[i].lstrip("-"), flags[i + 1]
            if k in ["error", "freq", "scale", "MJD", "flags", "obs"]:
                log.error("TOA flag ({0}) will overwrite TOA parameter!".format(k))
                raise (ValueError)
            try:  # Convert what we can to floats and ints
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
    elif fmt == "Command":
        d[fmt] = line.split()
    elif fmt == "Parkes":
        """
        columns     item
        1-1         Must be blank
        26-34       Observing Frequency (MHz)
        35-55       TOA (decimal point must be in column 42)
        56-63       Phase offset (fraction of P0, added to TOA)
        64-71       TOA uncertainty
        80-80       Observatory (1 character)
        """
        d["name"] = line[1:25]
        d["freq"] = float(line[25:34])
        ii = line[34:41]
        ff = line[42:55]
        MJD = (int(ii), float("0." + ff))
        phaseoffset = float(line[55:62])
        if phaseoffset != 0:
            raise ValueError(
                "Cannot interpret Parkes format with phaseoffset=%f yet" % phaseoffset
            )
        d["error"] = float(line[63:71])
        d["obs"] = get_observatory(line[79].upper()).name
    elif fmt == "ITOA":
        raise RuntimeError("TOA format '%s' not implemented yet" % fmt)
    return MJD, d


def format_toa_line(
    toatime,
    toaerr,
    freq,
    obs,
    dm=0.0 * u.pc / u.cm ** 3,
    name="unk",
    flags={},
    format="Princeton",
):
    """Format TOA line for writing

    Parameters
    ----------
    toatime
        Time object containing TOA arrival time
    toaerr
        TOA error as a Quantity with units
    freq
        Frequency as a Quantity with units (NB: value of np.inf is allowed)
    obs
        Observatory object
    dm
        DM for the TOA as a Quantity with units (not printed if 0.0 pc/cm^3)
    name
        Name to embed in TOA line (conventionally the data file name)
    format
        (Princeton | Tempo2)
    flags
        Any Tempo2 flags to append to the TOA line

    Returns
    -------
    out : string
        Formatted TOA line

    Note
    ----
    This implementation is currently incomplete in that it will not
    undo things like TIME statements and probably other things.

    Princeton format::

        columns  item
        1-1     Observatory (one-character code) '@' is barycenter
        2-2     must be blank
        16-24   Observing frequency (MHz)
        25-44   TOA (decimal point must be in column 30 or column 31)
        45-53   TOA uncertainty (microseconds)
        69-78   DM correction (pc cm^-3)

    Tempo2 format:

        - First line of file should be "``FORMAT 1``"
        - TOA format is ``file freq sat satErr siteID <flags>``

    """
    if format.upper() in ("TEMPO2", "1"):
        # This should probably use obs.timescale instaed of this hack
        if obs.tempo_code == "@":
            toa_str = Time(toatime, format="pulsar_mjd_string", scale="tdb")
        else:
            toa_str = Time(toatime, format="pulsar_mjd_string", scale="utc")
        # In Tempo2 format, freq=0.0 means infinite frequency
        if freq == np.inf * u.MHz:
            freq = 0.0 * u.MHz
        flagstring = ""
        if dm != 0.0 * u.pc / u.cm ** 3:
            flagstring += "-dm {0:%.5f}".format(dm.to(u.pc / u.cm ** 3).value)
        # Here I need to append any actual flags
        for flag in flags.keys():
            v = flags[flag]
            # Since toas file do not have values with unit in the flags,
            # here we are taking the units out
            if flag in ["clkcorr"]:
                continue
            if hasattr(v, "unit"):
                v = v.value
            flag = str(flag)
            if flag.startswith("-"):
                flagstring += " %s %s" % (flag, v)
            else:
                flagstring += " -%s %s" % (flag, v)
        # Now set observatory code. Use obs.name unless overridden by tempo2_code
        try:
            obscode = obs.tempo2_code
        except AttributeError:
            obscode = obs.name
        out = "%s %f %s %.3f %s %s\n" % (
            name,
            freq.to(u.MHz).value,
            toa_str,
            toaerr.to(u.us).value,
            obscode,
            flagstring,
        )
    elif format.upper() in ("PRINCETON", "TEMPO"):
        # This should probably use obs.timescale instead of this hack
        if obs.tempo_code == "@":
            toa_str = str(Time(toatime, format="pulsar_mjd_string", scale="tdb"))
        else:
            toa_str = str(Time(toatime, format="pulsar_mjd_string", scale="utc"))
        # The Princeton format can only deal with MJDs that have up to 20
        # digits, so truncate if longer.
        if len(toa_str) > 20:
            toa_str = toa_str[:20]
        # In TEMPO/Princeton format, freq=0.0 means infinite frequency
        if freq == np.inf * u.MHz:
            freq = 0.0 * u.MHz
        if obs.tempo_code is None:
            raise ValueError(
                "Observatory {} does not have 1-character tempo_code!".format(obs.name)
            )
        if dm != 0.0 * u.pc / u.cm ** 3:
            out = obs.tempo_code + " %13s%9.3f%20s%9.2f                %9.4f\n" % (
                name,
                freq.to(u.MHz).value,
                toa_str,
                toaerr.to(u.us).value,
                dm.to(u.pc / u.cm ** 3).value,
            )
        else:
            out = obs.tempo_code + " %13s%9.3f%20s%9.2f\n" % (
                name,
                freq.to(u.MHz).value,
                toa_str,
                toaerr.to(u.us).value,
            )
    else:
        raise ValueError("Unknown TOA format ({0})".format(format))

    return out


def make_fake_toas(startMJD, endMJD, ntoas, model, freq=1400, obs="GBT"):
    """Make evenly spaced toas with residuals = 0 and  without errors

    might be able to do different frequencies if fed an array of frequencies,
    only works with one observatory at a time

    Parameters
    ----------
    startMJD
        starting MJD for fake toas
    endMJD
        ending MJD for fake toas
    ntoas
        number of fake toas to create between startMJD and endMJD
    model
        current model
    freq : float, optional
        frequency of the fake toas, default 1400
    obs : str, optional
        observatory for fake toas, default GBT

    Returns
    -------
    TOAs
        object with evenly spaced toas spanning given start and end MJD with
        ntoas toas, without errors
    """
    # TODO:make all variables Quantity objects
    # TODO: freq default to inf
    def get_freq_array(bfv, ntoas):
        freq = np.zeros(ntoas)
        num_freqs = len(bfv)
        for ii, fv in enumerate(bfv):
            freq[ii::num_freqs] = fv
        return freq

    times = (
        np.linspace(np.longdouble(startMJD) * u.d, np.longdouble(endMJD) * u.d, ntoas)
        * u.day
    )
    freq_array = get_freq_array(np.atleast_1d(freq) * u.MHz, len(times))
    t1 = [
        TOA(t.value, obs=obs, freq=f, scale=get_observatory(obs).timescale)
        for t, f in zip(times, freq_array)
    ]
    ts = TOAs(toalist=t1)
    ts.compute_TDBs()
    ts.compute_posvels()
    ts.clock_corr_info.update(
        {"include_bipm": False, "bipm_version": "BIPM2015", "include_gps": False}
    )
    return ts


class TOA(object):
    """A time of arrival (TOA) class.

    This is a class for representing a single pulse arrival
    time measurement. It carries both the time - which needs careful handling
    as we often need more precision than python floats can provide - and
    a collection of additional data necessary to work with the data. These
    are often obtained by reading ``.tim`` files produced by pulsar data
    analysis software, but they can also be constructed as python objects.

    Parameters
    ----------
    MJD : astropy.time.Time, float, or tuple of floats
        The time of the TOA, which can be expressed as an astropy Time,
        a floating point MJD (64 or 80 bit precision), or a tuple
        of (MJD1,MJD2) whose sum is the full precision MJD (usually the
        integer and fractional part of the MJD)
    error : astropy.units.Quantity or float
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds
    obs : str
        The observatory code for the TOA
    freq : float or astropy.units.Quantity
        Frequency corresponding to the TOA.  Either a Quantity with frequency
        units, or a number for which MHz is assumed.
    scale : str
        Time scale for the TOA time.  Defaults to the timescale appropriate
        to the site, but can be overridden
    flags : dict
        Flags associated with the TOA.  If flags is not provided, any
        additional keyword arguments are interpreted as flags.

    Attributes
    ----------
    mjd : astropy.time.Time
        The pulse arrival time
    error : astropy.units.Quantity
        The uncertainty on the pulse arrival time
    obs : str
        The observatory code
    freq : astropy.units.Quantity
        The observing frequency
    flags : dict
        Any additional flags that were set for this TOA

    Notes
    -----
    MJDs will be stored in astropy.time.Time format, and can be
    passed as a double (not recommended), a string, a
    tuple of component parts (usually day and fraction of day).
    error is the TOA uncertainty in microseconds
    obs is the observatory name as defined by the Observatory class
    freq is the observatory-centric frequency in MHz
    other keyword/value pairs can be specified as needed

    It is VERY important that all astropy.Time() objects are created
    with precision=9. This is ensured in the code and is checked for any
    Time object passed to the TOA constructor.

    A discussion of times and clock corrections in PINT is available here:
    https://github.com/nanograv/PINT/wiki/Clock-Corrections-and-Timescales-in-PINT

    Observatory codes are (semi-)standardized short strings describing
    particular observatories. PINT needs to know considerable additional
    information about the observatory, including its precise position and
    clock correction details.

    Examples
    --------

    Constructing a TOA object::

        >>> a = TOA((54567, 0.876876876876876), 4.5, freq=1400.0,
        ...         obs="GBT", backend="GUPPI")
        >>> print a
        54567.876876876876876:  4.500 us error from 'GBT' at 1400.0000 MHz {'backend': 'GUPPI'}

    What happens if IERS data is not available for the date::

        >>> a = TOA((154567, 0.876876876876876), 4.5, freq=1400.0,
        ...         obs="GBT", backend="GUPPI")

        Traceback (most recent call last):
          omitted
        IndexError: (some) times are outside of range covered by IERS table.

    """

    def __init__(
        self,
        MJD,
        error=0.0,
        obs="Barycenter",
        freq=float("inf"),
        scale=None,
        flags=None,
        **kwargs
    ):
        site = get_observatory(obs)
        # If MJD is already a Time, just use it. Note that this will ignore
        # the 'scale' argument to the TOA() constructor!
        if isinstance(MJD, time.Time):
            if scale is not None:
                raise ValueError("scale argument is ignored when Time is provided")
            t = MJD
        else:
            try:
                arg1, arg2 = MJD
            except TypeError:
                arg1, arg2 = MJD, None
            if scale is None:
                scale = site.timescale
            # First build a time without a location
            # Note that when scale is UTC, must use pulsar_mjd format!
            if scale.lower() == "utc":
                fmt = "pulsar_mjd"
            else:
                fmt = "mjd"
            t = time.Time(arg1, arg2, scale=scale, format=fmt, precision=9)

        # Now assign the site location to the Time, for use in the TDB conversion
        # Time objects are immutable so you must make a new one to add the location!
        # Use the intial time to look up the observatory location
        # (needed for moving observatories)
        # The location is an EarthLocation in the ITRF (ECEF, WGS84) frame
        try:
            loc = site.earth_location_itrf(time=t)
        except Exception:
            # Just add informmation and re-raise
            log.error(
                "Error computing earth_location_itrf at time {0}, {1}".format(
                    t, type(t)
                )
            )
            raise
        # Then construct the full time, with observatory location set
        self.mjd = time.Time(t, location=loc, precision=9)

        if hasattr(error, "unit"):
            try:
                self.error = error.to(u.microsecond)
            except u.UnitConversionError:
                raise u.UnitConversionError(
                    "Uncertainty for TOA with incompatible unit {0}".format(error)
                )
        else:
            self.error = error * u.microsecond
        self.obs = site.name
        if hasattr(freq, "unit"):
            try:
                self.freq = freq.to(u.MHz)
            except u.UnitConversionError:
                raise u.UnitConversionError(
                    "Frequency for TOA with incompatible unit {0}".format(freq)
                )
        else:
            self.freq = freq * u.MHz
        if self.freq == 0.0 * u.MHz:
            self.freq = np.inf * u.MHz
        if flags is None:
            self.flags = kwargs
        else:
            self.flags = flags
            if kwargs:
                raise TypeError(
                    "TOA constructor does not accept keyword arguments {}".format(
                        kwargs
                    )
                )

    def __str__(self):
        s = self.mjd.mjd_string + ": %6.3f %s error from '%s' at %.4f %s " % (
            self.error.value,
            self.error.unit,
            self.obs,
            self.freq.value,
            self.freq.unit,
        )
        if self.flags:
            s += str(self.flags)
        return s

    def as_line(self, format="Tempo2", name="_", dm=0 * u.pc / u.cm ** 3):
        return format_toa_line(
            mjd=self.mjd,
            error=self.error,
            freq=self.freq,
            obs=self.obs,
            dm=dm,
            name=name,
            format=format,
            flags=self.flags,
        )


class TOAs(object):
    """A class of multiple TOAs, loaded from zero or more files.

    The contents are stored in an `astropy.table.Table`

    Parameters
    ----------
    toafile : str, optional
        Filename to load TOAs from.
    toalist : list of TOA objects, optional
        The TOA objects this TOAs should contain.  Exactly one of
        these two parameters must be provided.

    Attributes
    ----------
    table : astropy.table.Table
        The data for all the TOAs is stored in here. It has the columns
        ``index`` (the location of the TOA in the original input),
        ``mjd`` (an :class:`astropy.time.Time` object), ``mjd_float`` (a
        floating-point version of the time), ``error`` (an
        :class:`astropy.units.Quantity` describing the claimed uncertainty
        on the pulse arrival time), ``freq`` (an :class:`astropy.units.Quantity`
        describing the observing frequency), ``obs`` (a
        :class:`pint.observatory.Observatory` object),
        and ``flags`` (a dictionary of flags and their values). The table may
        also contain a column ``pn`` (integers) that is the pulse numbers
        of the TOAs.  The table is grouped by ``obs``, that is, it is
        not in the same order as the original TOAs.
    commands : list of str
        "Commands" that were written in the file; these can affect
        how some or all TOAs are interpreted.
    filename : str, optional
        The filename (if any) that the TOAs were loaded from.
    planets : bool
    ephem : object
    clock_corr_info : dict

    """

    def __init__(self, toafile=None, toalist=None):
        # First, just make an empty container
        self.toas = []
        self.commands = []
        self.filename = None
        self.planets = False
        self.ephem = None
        self.clock_corr_info = {}

        if (toalist is not None) and (toafile is not None):
            raise ValueError("Cannot initialize TOAs from both file and list.")

        if toafile is not None:
            # FIXME: work with file-like objects as well
            # Check for a pickle-like filename.  Alternative approach would
            # be to just try opening it as a pickle and see what happens.
            if toafile.endswith(".pickle") or toafile.endswith("pickle.gz"):
                log.info("Reading TOAs from pickle file")
                self.read_pickle_file(toafile)
            else:
                self.read_toa_file(toafile)
                self.filename = toafile

        if toalist is not None:
            if not isinstance(toalist, (list, tuple)):
                raise ValueError("Trying to initialize TOAs from a non-list class")
            self.toas = toalist

        if not hasattr(self, "table"):
            mjds = self.get_mjds(high_precision=True)
            # The table is grouped by observatory
            self.table = table.Table(
                [
                    np.arange(len(mjds)),
                    table.Column(mjds),
                    self.get_mjds(),
                    self.get_errors(),
                    self.get_freqs(),
                    self.get_obss(),
                    self.get_flags(),
                    np.zeros(len(mjds)),
                    self.get_groups(),
                ],
                names=(
                    "index",
                    "mjd",
                    "mjd_float",
                    "error",
                    "freq",
                    "obs",
                    "flags",
                    "delta_pulse_number",
                    "groups",
                ),
                meta={"filename": self.filename},
            ).group_by("obs")
            # Add pulse number column (if needed) or make PHASE adjustments
            try:
                self.phase_columns_from_flags()
            except ValueError:
                log.debug("No pulse numbers found in the TOAs")

        # We don't need this now that we have a table
        del self.toas

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
            return np.array([t.freq.to(u.MHz).value for t in self.toas]) * u.MHz
        else:
            return self.table["freq"].quantity

    def get_mjds(self, high_precision=False):
        """Array of MJDs in the TOAs object

        With high_precision is True
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
                return np.array([t.mjd for t in self.toas])
            else:
                return np.array([t for t in self.table["mjd"]])
        else:
            if hasattr(self, "toas"):
                return np.array([t.mjd.mjd for t in self.toas]) * u.day
            else:
                return self.table["mjd_float"].quantity

    def get_errors(self):
        """Return a numpy array of the TOA errors in us"""
        # FIXME temporarily disable reading errors from toas
        if hasattr(self, "toas"):
            return np.array([t.error.to(u.us).value for t in self.toas]) * u.us
        else:
            return self.table["error"].quantity

    def get_obss(self):
        """Return a numpy array of the observatories for each TOA"""
        if hasattr(self, "toas"):
            return np.array([t.obs for t in self.toas])
        else:
            return self.table["obs"]

    def get_pulse_numbers(self):
        """Return a numpy array of the pulse numbers for each TOA if they exist"""
        # TODO: use a masked array?  Only some pulse numbers may be known
        if hasattr(self, "toas"):
            try:
                return np.array([t.flags["pn"] for t in self.toas])
            except KeyError:
                log.warning("Not all TOAs have pulse numbers, using none")
                return None
        else:
            if "pn" in self.table["flags"][0]:
                if "pulse_number" in self.table.colnames:
                    raise ValueError(
                        "Pulse number cannot be both a column and a TOA flag"
                    )
                return np.array(flags["pn"] for flags in self.table["flags"])
            elif "pulse_number" in self.table.colnames:
                return self.table["pulse_number"]
            else:
                log.warning("No pulse numbers for TOAs")
                return None

    def get_flags(self):
        """Return a numpy array of the TOA flags"""
        if hasattr(self, "toas"):
            return np.array([t.flags for t in self.toas])
        else:
            return self.table["flags"]

    def get_flag_value(self, flag, fill_value=None):
        """Get the request TOA flag values.

           Parameters
           ----------
           flag_name : str
               The request flag name.

           Returns
           -------
           values : list
               A list of flag values from each TOA. If the TOA does not have
               the flag, it will fill up with the fill_value.
        """
        result = []
        for flags in self.table["flags"]:
            val = flags.get(flag, fill_value)
            result.append(val)
        return result

    def get_groups(self, gap_limit=None):
        """flag toas within gap limit (default 2h = 0.0833d) of each other as the same group

        groups can be larger than the gap limit - if toas are seperated by a gap larger than
        the gap limit, a new group starts and continues until another such gap is found"""
        # TODO: make all values Quantity objects for consistency
        if gap_limit == None:
            gap_limit = 0.0833
        if hasattr(self, "toas") or gap_limit != 0.0833:
            gap_limit *= u.d
            mjd_dict = OrderedDict()
            mjd_values = self.get_mjds().value
            for i in np.arange(len(mjd_values)):
                mjd_dict[i] = mjd_values[i]
            sorted_mjd_list = sorted(mjd_dict.items(), key=lambda kv: (kv[1], kv[0]))
            indexes = [a[0] for a in sorted_mjd_list]
            mjds = [a[1] for a in sorted_mjd_list]
            gaps = np.diff(mjds)
            lengths = []
            count = 0
            for i in range(len(gaps)):
                if gaps[i] * u.d < gap_limit:
                    count += 1
                else:
                    lengths += [count + 1]
                    count = 0
            lengths += [count + 1]
            sorted_groups = []
            groupnum = 0
            for length in lengths:
                sorted_groups += [groupnum] * length
                groupnum += 1
            group_dict = OrderedDict()
            for i in np.arange(len(indexes)):
                group_dict[indexes[i]] = sorted_groups[i]
            groups = [group_dict[key] for key in sorted(group_dict)]
            return groups
        else:
            return self.table["groups"]

    def get_highest_density_range(self, ndays=7):
        """print the range of mjds (default 7 days) with the most toas"""
        # TODO: implement sliding window
        nbins = int((max(self.get_mjds()) - min(self.get_mjds())) / (ndays * u.d))
        a = np.histogram(self.get_mjds(), nbins)
        maxday = int(a[1][np.argmax(a[0])])
        diff = int(a[1][1] - a[1][0])
        print(
            "max density range (in steps of {} days -- {} bins) is from MJD {} to {} with {} toas.".format(
                diff, nbins, maxday, maxday + diff, a[0].max()
            )
        )
        return (maxday, maxday + diff)

    def select(self, selectarray):
        """Apply a boolean selection or mask array to the TOA table.

        This operation modifies the TOAs object in place, shrinking its
        table down to just those TOAs where selectarray is True. This
        function also stores the old table in a stack.
        """
        if hasattr(self, "table"):
            # Allow for selection undos
            if not hasattr(self, "table_selects"):
                self.table_selects = []
            self.table_selects.append(copy.deepcopy(self.table))
            # Our TOA table must be grouped by observatory for phase calcs
            self.table = self.table[selectarray].group_by("obs")
        else:
            raise ValueError("TOA selection not implemented for TOA lists.")

    def unselect(self):
        """Return to previous selected version of the TOA table (stored in stack)."""
        try:
            self.table = self.table_selects.pop()
        except (AttributeError, IndexError) as e:
            log.error("No previous TOA table found.  No changes made.")

    def pickle(self, filename=None):
        """Write the TOAs to a .pickle file with optional filename."""
        if filename is not None:
            pickle.dump(self, open(filename, "wb"))
        elif self.filename is not None:
            pickle.dump(self, gzip.open(self.filename + ".pickle.gz", "wb"))
        else:
            raise ValueError("TOA pickle method needs a filename.")

    def get_summary(self):
        """Return a short ASCII summary of the TOAs."""
        s = "Number of TOAs:  %d\n" % self.ntoas
        s += "Number of commands:  %d\n" % len(self.commands)
        s += "Number of observatories:  %d %s\n" % (
            len(self.observatories),
            list(self.observatories),
        )
        s += "MJD span:  %.3f to %.3f\n" % (self.first_MJD.mjd, self.last_MJD.mjd)
        s += "Date span: {} to {}\n".format(self.first_MJD.iso, self.last_MJD.iso)
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            s += "%s TOAs (%d):\n" % (key["obs"], len(grp))
            s += "  Min freq:      {:.3f} \n".format(np.min(grp["freq"].to(u.MHz)))
            s += "  Max freq:      {:.3f} \n".format(np.max(grp["freq"].to(u.MHz)))
            s += "  Min error:     {:.3g}\n".format(np.min(grp["error"].to(u.us)))
            s += "  Max error:     {:.3g}\n".format(np.max(grp["error"].to(u.us)))
            s += "  Median error:  {:.3g}\n".format(np.median(grp["error"].to(u.us)))
        return s

    def print_summary(self):
        """Write a summary of the TOAs to stdout."""
        print(self.get_summary())

    def phase_columns_from_flags(self):
        """Creates and/or modifies pulse_number and delta_pulse_number columns

        Scans pulse numbers from the table flags and creates a new table column.
        Modifes the delta_pulse_number column, if required.
        Removes the pulse numbers from the flags.
        """
        # Add pulse_number as a table column if possible
        try:
            pns = [flags["pn"] for flags in self.table["flags"]]
            self.table["pulse_number"] = pns
            self.table["pulse_number"].unit = u.dimensionless_unscaled

            # Remove pn from dictionary to prevent redundancies
            for flags in self.table["flags"]:
                del flags["pn"]
        except KeyError:
            raise ValueError("Not all TOAs have pn flags")
        # modify the delta_pulse_number column if required
        dphs = np.asarray(
            [
                flags["phase"] if "phase" in flags else 0.0
                for flags in self.table["flags"]
            ]
        )
        self.table["delta_pulse_number"] += dphs

    def compute_pulse_numbers(self, model):
        """Set pulse numbers (in TOA table column pulse_numbers) based on model

        Replace any existing pulse numbers by computing phases according to
        model and then setting the pulse number of each to their integer part,
        which the nearest integer since Phase objects ensure that.
        """
        # paulr: I think pulse numbers should be computed with abs_phase=True!
        phases = model.phase(self, abs_phase=True)
        self.table["pulse_number"] = phases.int
        self.table["pulse_number"].unit = u.dimensionless_unscaled

    def adjust_TOAs(self, delta):
        """Apply a time delta to TOAs

        Adjusts the time (MJD) of the TOAs by applying delta, which should
        have the same shape as ``self.table['mjd']``.  This function does not change
        the pulse numbers column, if present, but does recompute ``mjd_float``,
        the TDB times, and the observatory positions and velocities.

        Parameters
        ----------
        delta : astropy.time.TimeDelta
            The time difference to add to the MJD of each TOA

        """
        col = self.table["mjd"]
        if not isinstance(delta, time.TimeDelta):
            raise ValueError("Type of argument must be TimeDelta")
        if delta.shape != col.shape:
            raise ValueError("Shape of mjd column and delta must be compatible")
        for ii in range(len(col)):
            col[ii] = col[ii] + delta[ii]

        # This adjustment invalidates the derived columns in the table, so delete
        # and recompute them
        # Changed high_precision from False to True to avoid self referential get_mjds()
        self.table["mjd_float"] = [
            t.mjd for t in self.get_mjds(high_precision=True)
        ] * u.day
        self.compute_TDBs()
        self.compute_posvels(self.ephem, self.planets)

    def write_TOA_file(self, filename, name="pint", format="Princeton"):
        """Dump current TOA table out as a TOA file

        Parameters
        ----------
        filename : str or file-like
            File name to write to; can be an open file object
        format : str
            Format specifier for file ('TEMPO' or 'Princeton') or ('Tempo2' or '1')

        """
        try:
            # FIXME: file must be closed even if an exception occurs!
            # Answer is to use a with statement and call the function recursively
            outf = open(filename, "w")
            handle = False
        except TypeError:
            outf = filename
            handle = True
        if format.upper() in ("TEMPO2", "1"):
            outf.write("FORMAT 1\n")

        # Add pulse numbers to flags temporarily if there is a pulse number column
        pnChange = False
        if "pn" in self.table.colnames:
            pnChange = True
            for i in range(len(self.table["flags"])):
                self.table["flags"][i]["pn"] = self.table["pn"][i]

        for (toatime, toaerr, freq, obs, flags) in zip(
            self.table["mjd"],
            self.table["error"].quantity,
            self.table["freq"].quantity,
            self.table["obs"],
            self.table["flags"],
        ):
            obs_obj = Observatory.get(obs)

            if "clkcorr" in flags.keys():
                toatime_out = toatime - time.TimeDelta(flags["clkcorr"])
            else:
                toatime_out = toatime
            out_str = format_toa_line(
                toatime_out,
                toaerr,
                freq,
                obs_obj,
                name=name,
                flags=flags,
                format=format,
            )
            outf.write(out_str)

        # If pulse numbers were added to flags, remove them again
        if pnChange:
            for flags in self.table["flags"]:
                del flags["pn"]

        if not handle:
            outf.close()

    def apply_clock_corrections(
        self, include_bipm=True, bipm_version="BIPM2015", include_gps=True
    ):
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
        flags = self.table["flags"]
        if any(["clkcorr" in f for f in flags]):
            if all(["clkcorr" in f for f in flags]):
                log.warning("Clock corrections already applied. Not re-applying.")
                return
            else:
                # FIXME: could apply clock corrections to just the ones that don't have any
                raise ValueError("Some TOAs have 'clkcorr' flag and some do not!")
        # An array of all the time corrections, one for each TOA
        log.info(
            "Applying clock corrections (include_GPS = {0}, include_BIPM = {1})".format(
                include_gps, include_bipm
            )
        )
        corr = np.zeros(self.ntoas) * u.s
        times = self.table["mjd"]
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]["obs"]
            site = get_observatory(
                obs,
                include_gps=include_gps,
                include_bipm=include_bipm,
                bipm_version=bipm_version,
            )
            loind, hiind = self.table.groups.indices[ii : ii + 2]
            # First apply any TIME statements
            for jj in range(loind, hiind):
                if "to" in flags[jj]:
                    # TIME commands are in sec
                    # SUGGESTION(@paulray): These time correction units should
                    # be applied in the parser, not here. In the table the time
                    # correction should have units.
                    corr[jj] = flags[jj]["to"] * u.s
                    times[jj] += time.TimeDelta(corr[jj])

            gcorr = site.clock_corrections(time.Time(grp["mjd"]))
            for jj, cc in enumerate(gcorr):
                grp["mjd"][jj] += time.TimeDelta(cc)
            corr[loind:hiind] += gcorr
            # Now update the flags with the clock correction used
            for jj in range(loind, hiind):
                if corr[jj] != 0:
                    flags[jj]["clkcorr"] = corr[jj]
        # Update clock correction info
        self.clock_corr_info.update(
            {
                "include_bipm": include_bipm,
                "bipm_version": bipm_version,
                "include_gps": include_gps,
            }
        )

    def compute_TDBs(self, method="default", ephem=None):
        """Compute and add TDB and TDB long double columns to the TOA table.

        This routine creates new columns 'tdb' and 'tdbld' in a TOA table
        for TDB times, using the Observatory locations and IERS A Earth
        rotation corrections for UT1.

        """
        log.info("Computing TDB columns.")
        if "tdb" in self.table.colnames:
            log.info("tdb column already exists. Deleting...")
            self.table.remove_column("tdb")
        if "tdbld" in self.table.colnames:
            log.info("tdbld column already exists. Deleting...")
            self.table.remove_column("tdbld")

        if ephem is None:
            if self.ephem is not None:
                ephem = self.ephem
            else:
                log.warning(
                    "No ephemeris provided to TOAs object or compute_TDBs. Using DE421"
                )
                ephem = "DE421"
        else:
            # If user specifies an ephemeris, make sure it is the same as the one already
            # in the TOA object, to prevent mixing.
            if (self.ephem is not None) and (ephem != self.ephem):
                log.error(
                    "Ephemeris provided to compute_TDBs {0} is different than TOAs object "
                    "ephemeris {1}! Using TDB ephemeris.".format(ephem, self.ephem)
                )
        self.ephem = ephem

        # Compute in observatory groups
        tdbs = np.zeros_like(self.table["mjd"])
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]["obs"]
            loind, hiind = self.table.groups.indices[ii : ii + 2]
            site = get_observatory(obs)
            if isinstance(site, TopoObs):
                # For TopoObs, it is safe to assume that all TOAs have same location
                # I think we should report to astropy that initializing
                # a Time from a list (or Column) of Times throws away the location information
                grpmjds = time.Time(grp["mjd"], location=grp["mjd"][0].location)
            else:
                # Grab locations for each TOA
                # It is crazy that I have to deconstruct the locations like
                # this to build a single EarthLocation object with an array
                # of locations contained in it.
                # Is there a more efficient way to convert a list of EarthLocations
                # into a single EarthLocation object with an array of values internally?
                loclist = [t.location for t in grp["mjd"]]
                if loclist[0] is None:
                    grpmjds = time.Time(grp["mjd"], location=None)
                else:
                    locs = EarthLocation(
                        np.array([l.x.value for l in loclist]) * u.m,
                        np.array([l.y.value for l in loclist]) * u.m,
                        np.array([l.z.value for l in loclist]) * u.m,
                    )
                    grpmjds = time.Time(grp["mjd"], location=locs)

            if isinstance(site, SpacecraftObs):
                grptdbs = site.get_TDBs(grpmjds, method=method, ephem=ephem, grp=grp)
            else:
                grptdbs = site.get_TDBs(grpmjds, method=method, ephem=ephem)
            tdbs[loind:hiind] = np.asarray([t for t in grptdbs])

        # Now add the new columns to the table
        col_tdb = table.Column(name="tdb", data=tdbs)
        col_tdbld = table.Column(name="tdbld", data=[t.tdb.mjd_long for t in tdbs])
        self.table.add_columns([col_tdb, col_tdbld])

    def compute_posvels(self, ephem=None, planets=False):
        """Compute positions and velocities of the observatories and Earth.

        Compute the positions and velocities of the observatory (wrt
        the Geocenter) and the center of the Earth (referenced to the
        SSB) for each TOA.  The JPL solar system ephemeris can be set
        using the 'ephem' parameter.  The positions and velocities are
        set with PosVel class instances which have astropy units.

        """

        if ephem is None:
            if self.ephem is not None:
                ephem = self.ephem
            else:
                log.warning(
                    "No ephemeris provided to TOAs object or compute_posvels. Using DE421"
                )
                ephem = "DE421"
        else:
            # If user specifies an ephemeris, make sure it is the same as the one already in
            # the TOA object, to prevent mixing.
            if (self.ephem is not None) and (ephem != self.ephem):
                log.error(
                    "Ephemeris provided to compute_posvels {0} is different than "
                    "TOAs object ephemeris {1}! Using posvels ephemeris.".format(
                        ephem, self.ephem
                    )
                )
        # Record the choice of ephemeris and planets
        self.ephem = ephem
        self.planets = planets
        if planets:
            log.info(
                "Computing PosVels of observatories, Earth and planets, using {}".format(
                    ephem
                )
            )

        else:
            log.info(
                "Computing PosVels of observatories and Earth, using {}".format(ephem)
            )
        # Remove any existing columns
        cols_to_remove = ["ssb_obs_pos", "ssb_obs_vel", "obs_sun_pos"]
        for c in cols_to_remove:
            if c in self.table.colnames:
                log.info("Column {0} already exists. Removing...".format(c))
                self.table.remove_column(c)
        for p in ("jupiter", "saturn", "venus", "uranus"):
            name = "obs_" + p + "_pos"
            if name in self.table.colnames:
                log.info("Column {0} already exists. Removing...".format(name))
                self.table.remove_column(name)

        self.table.meta["ephem"] = ephem
        ssb_obs_pos = table.Column(
            name="ssb_obs_pos",
            data=np.zeros((self.ntoas, 3), dtype=np.float64),
            unit=u.km,
            meta={"origin": "SSB", "obj": "OBS"},
        )
        ssb_obs_vel = table.Column(
            name="ssb_obs_vel",
            data=np.zeros((self.ntoas, 3), dtype=np.float64),
            unit=u.km / u.s,
            meta={"origin": "SSB", "obj": "OBS"},
        )
        obs_sun_pos = table.Column(
            name="obs_sun_pos",
            data=np.zeros((self.ntoas, 3), dtype=np.float64),
            unit=u.km,
            meta={"origin": "OBS", "obj": "SUN"},
        )
        if planets:
            plan_poss = {}
            for p in ("jupiter", "saturn", "venus", "uranus"):
                name = "obs_" + p + "_pos"
                plan_poss[name] = table.Column(
                    name=name,
                    data=np.zeros((self.ntoas, 3), dtype=np.float64),
                    unit=u.km,
                    meta={"origin": "OBS", "obj": p},
                )

        # Now step through in observatory groups
        for ii, key in enumerate(self.table.groups.keys):
            grp = self.table.groups[ii]
            obs = self.table.groups.keys[ii]["obs"]
            loind, hiind = self.table.groups.indices[ii : ii + 2]
            site = get_observatory(obs)
            tdb = time.Time(grp["tdb"], precision=9)

            if isinstance(site, SpacecraftObs):
                ssb_obs = site.posvel(tdb, ephem, grp)
            else:
                ssb_obs = site.posvel(tdb, ephem)

            log.debug("SSB obs pos {0}".format(ssb_obs.pos[:, 0]))
            ssb_obs_pos[loind:hiind, :] = ssb_obs.pos.T.to(u.km)
            ssb_obs_vel[loind:hiind, :] = ssb_obs.vel.T.to(u.km / u.s)
            sun_obs = objPosVel_wrt_SSB("sun", tdb, ephem) - ssb_obs
            obs_sun_pos[loind:hiind, :] = sun_obs.pos.T.to(u.km)
            if planets:
                for p in ("jupiter", "saturn", "venus", "uranus"):
                    name = "obs_" + p + "_pos"
                    dest = p
                    pv = objPosVel_wrt_SSB(dest, tdb, ephem) - ssb_obs
                    plan_poss[name][loind:hiind, :] = pv.pos.T.to(u.km)
        cols_to_add = [ssb_obs_pos, ssb_obs_vel, obs_sun_pos]
        if planets:
            cols_to_add += plan_poss.values()
        log.debug("Adding columns " + " ".join([cc.name for cc in cols_to_add]))
        self.table.add_columns(cols_to_add)

    def read_pickle_file(self, filename):
        """Read the TOAs from the pickle file specified in filename.

        Note the filename should include any pickle-specific extensions (ie
        ".pickle.gz" or similar), these will not be added automatically.

        If the file ends with ".gz" it will be uncompressed before extracting
        the pickle.
        """

        log.info("Reading pickled TOAs from '%s'..." % filename)
        if os.path.splitext(filename)[1] == ".gz":
            infile = gzip.open(filename, "rb")
        else:
            infile = open(filename, "rb")
        tmp = pickle.load(infile)
        self.filename = tmp.filename
        if hasattr(tmp, "toas"):
            self.toas = tmp.toas
        if hasattr(tmp, "table"):
            self.table = tmp.table.group_by("obs")
        self.commands = tmp.commands

    def read_toa_file(self, filename, process_includes=True, top=True):
        """Read TOAs from the given filename.

        Will process INCLUDEd files unless process_includes is False.

        Parameters
        ----------
        filename : str
            The name of the file to open.
        process_includes : bool, optional
            If true, obey INCLUDE directives in the file and read other
            files.
        top : bool, optional
            If true, wipe this instance's contents, otherwise append
            new TOAs. Used recursively; note that surprises may ensue
            if this function is called on an already existing and
            processed TOAs object.
        """
        ntoas = 0
        if top:
            self.toas = []
            self.commands = []
            self.cdict = {
                "EFAC": 1.0,
                "EQUAD": 0.0 * u.us,
                "EMIN": 0.0 * u.us,
                "EMAX": np.inf * u.us,
                "FMIN": 0.0 * u.MHz,
                "FMAX": np.inf * u.MHz,
                "INFO": None,
                "SKIP": False,
                "TIME": 0.0,
                "PHASE": 0,
                "PHA1": None,
                "PHA2": None,
                "MODE": 1,
                "JUMP": [False, 0],
                "FORMAT": "Unknown",
                "END": False,
            }
        with open(filename, "r") as f:
            for l in f.readlines():
                MJD, d = _parse_TOA_line(l, fmt=self.cdict["FORMAT"])
                if d["format"] == "Command":
                    cmd = d["Command"][0].upper()
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
                    elif cmd in ("EMIN", "EMAX", "EQUAD"):
                        self.cdict[cmd] = float(d["Command"][1]) * u.us
                    elif cmd in ("FMIN", "FMAX", "EQUAD"):
                        self.cdict[cmd] = float(d["Command"][1]) * u.MHz
                    elif cmd in ("EFAC", "PHA1", "PHA2"):
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
                        log.info(
                            "Processing included TOA file {0}".format(d["Command"][1])
                        )
                        self.read_toa_file(d["Command"][1], top=False)
                        # re-set FORMAT
                        self.cdict["FORMAT"] = fmt
                    else:
                        continue
                if self.cdict["SKIP"] or d["format"] in (
                    "Blank",
                    "Unknown",
                    "Comment",
                    "Command",
                ):
                    continue
                elif self.cdict["END"]:
                    if top:
                        # Clean up our temporaries used when reading TOAs
                        del self.cdict
                    return
                else:
                    newtoa = TOA(MJD, **d)
                    if (
                        (self.cdict["EMIN"] > newtoa.error)
                        or (self.cdict["EMAX"] < newtoa.error)
                        or (self.cdict["FMIN"] > newtoa.freq)
                        or (self.cdict["FMAX"] < newtoa.freq)
                    ):
                        continue
                    else:
                        newtoa.error *= self.cdict["EFAC"]
                        newtoa.error = np.hypot(newtoa.error, self.cdict["EQUAD"])
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
