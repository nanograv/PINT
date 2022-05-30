"""Routines for reading various formats of clock file."""

import os
import warnings
from textwrap import dedent

import astropy.units as u
import numpy as np
from loguru import logger as log

try:
    from erfa import ErfaWarning
except ImportError:
    from astropy._erfa import ErfaWarning

from pint.pulsar_mjd import Time
from pint.utils import open_or_use, lines_of


class ClockFileMeta(type):
    """Metaclass that provides a registry for different clock file formats.

    ClockFile implementations should define a 'format' class member giving
    the name of the format.
    """

    def __init__(cls, name, bases, members):
        regname = "_formats"
        if not hasattr(cls, regname):
            setattr(cls, regname, {})
        if "format" in members:
            getattr(cls, regname)[cls.format] = cls
        super(ClockFileMeta, cls).__init__(name, bases, members)


class ClockFile(metaclass=ClockFileMeta):
    """A clock correction file in one of several formats.

    The ClockFile class provides a way to read various formats of clock
    files.  It will provide the clock information from the file as arrays
    of times and clock correction values via the ClockFile.time and
    ClockFile.clock properties.  The file should be initially read using the
    ClockFile.read() method, for example:

        >>> cf = ClockFile.read(os.getenv('TEMPO')+'/clock/time_gbt.dat')
        >>> print cf.time
        [ 51909.5  51910.5  51911.5 ...,  57475.5  57476.5  57477.5]
        >>> print cf.clock
        [-3.14  -3.139 -3.152 ...,  0.179  0.185  0.188] us

    Or:

        >>> cf = ClockFile.read(os.getenv('TEMPO2')+'/clock/gbt2gps.clk',
                                    format='tempo2')
        >>> print cf.time
        [ 51909.5  51910.5  51911.5 ...,  57411.5  57412.5  57413.5]
        >>> print cf.clock
        [ -3.14000000e-06  -3.13900000e-06  -3.15200000e-06 ...,   1.80000000e-08
           2.10000000e-08   2.30000000e-08] s

    """

    @classmethod
    def read(cls, filename, format="tempo", **kwargs):
        if format in cls._formats.keys():
            r = cls._formats[format](filename, **kwargs)
            if not np.all(np.diff(r.time.mjd) >= 0):
                raise ValueError(
                    f"Clock file {filename} in format {format} appears to be out of order"
                )
            return r
        else:
            raise ValueError("clock file format '%s' not defined" % format)

    @property
    def time(self):
        return self._time

    @property
    def clock(self):
        return self._clock

    def evaluate(self, t, limits="warn"):
        """Evaluate the clock corrections at the times t.

        By default, values are linearly
        interpolated but this could be overridden by derived classes
        if needed.  The first/last values will be applied to times outside
        the data range.  If limits=='warn' this will also issue a warning.
        If limits=='error' an exception will be raised.

        Parameters
        ----------
        t : astropy.time.Time
            An array-valued Time object specifying the times at which to evaluate the
            clock correction.
        limits : "warn" or "error"
            If "error", raise an exception if times outside the range in the clock
            file are presented (or if the clock file is empty); if "warn", extrapolate
            by returning the value at the nearest endpoint but emit a warning.

        Returns
        -------
        corrections : astropy.units.Quantity
            The corrections in units of microseconds.
        """

        if len(self.time) == 0:
            msg = f"No data points in clock file '{self.filename}'"
            if limits == "warn":
                log.warning(msg)
                return np.zeros_like(t) * u.us
            elif limits == "error":
                raise RuntimeError(msg)

        if np.any(t < self.time[0]) or np.any(t > self.time[-1]):
            msg = f"Data points out of range in clock file '{self.filename}'"
            if limits == "warn":
                log.warning(msg)
            elif limits == "error":
                raise RuntimeError(msg)

        # Can't pass Times directly to np.interp.  This should be OK:
        return np.interp(t.mjd, self.time.mjd, self.clock.to(u.us).value) * u.us

    def last_correction_mjd(self):
        if len(self.time) == 0:
            return -np.inf
        else:
            return self.time[-1].mjd

    @staticmethod
    def merge(clocks, *, trim=True):
        """Compute clock correction information for a combination of clocks.

        Note that clock correction files can specify discontinuities by
        repeating MJDs. This merged object should accurately propagate these
        discontinuities.

        Parameters
        ----------
        trim : bool
            Whether to trim the resulting clock file to the MJD range
            covered by both input clock files.

        Returns
        -------
        ClockFile
            A merged ClockFile object.
        """
        all_mjds = []
        all_discontinuities = set()
        for c in clocks:
            mjds = c._time.mjd
            all_mjds.append(mjds)
            all_discontinuities.update(mjds[:-1][np.diff(mjds) == 0])
        mjds = np.unique(np.concatenate(all_mjds))
        r = np.ones(len(mjds), dtype=int)
        for m in all_discontinuities:
            i = np.searchsorted(mjds, m)
            r[i] = 2
        mjds = np.repeat(mjds, r)

        times = Time(mjds, format="pulsar_mjd", scale="utc")
        corr = np.zeros(len(mjds)) * u.s
        for c in clocks:
            # Interpolate everywhere
            this_corr = c.evaluate(times)
            # Find locations of left sides of discontinuities
            z = np.diff(c._time.mjd) == 0
            # Looking for the left end of a run of equal values
            zl = z.copy()
            zl[1:] &= ~z[:-1]
            ixl = np.where(zl)[0]
            # Fix discontinuities
            this_corr[np.searchsorted(mjds, c._time.mjd[ixl], side="left")] = c._clock[
                ixl
            ]

            zr = z.copy()
            zr[:-1] &= ~z[1:]
            ixr = np.where(zr)[0]
            # Fix discontinuities
            this_corr[np.searchsorted(mjds, c._time.mjd[ixr], side="right")] = c._clock[
                ixr + 1
            ]
            corr += this_corr
        if trim:
            b = max([c._time.mjd[0] for c in clocks])
            e = min([c._time.mjd[-1] for c in clocks])
            l = np.searchsorted(times.mjd, b)
            r = np.searchsorted(times.mjd, e, side="right")
            times = times[l:r]
            corr = corr[l:r]
        return ConstructedClockFile(
            mjd=times.mjd,
            clock=corr,
            filename=f"Merged from {[c.filename for c in clocks]}",
        )


class ConstructedClockFile(ClockFile):

    # No format set because these can't be read

    def __init__(self, mjd, clock, filename=None, **kwargs):
        if len(mjd) != len(clock):
            raise ValueError(f"MJDs have {len(mjd)} entries but clock has {len(clock)}")
        self._time = Time(mjd, format="pulsar_mjd", scale="utc")
        self._clock = clock.to(u.us)
        self.filename = "Constructed" if filename is None else filename


class Tempo2ClockFile(ClockFile):

    format = "tempo2"

    def __init__(self, filename, bogus_last_correction=False, **kwargs):
        self.filename = filename
        log.debug(f"Loading {self.format} observatory clock correction file {filename}")
        try:
            mjd, clk, self.header = self.load_tempo2_clock_file(filename)
        except (FileNotFoundError, OSError):
            log.error(f"TEMPO2-style clock correction file {filename} not found")
            mjd = np.array([], dtype=float)
            clk = np.array([], dtype=float)
            self.header = None
        if bogus_last_correction and len(mjd):
            mjd = mjd[:-1]
            clk = clk[:-1]
        while len(mjd) and mjd[0] == 0:
            # Zap leading zeros
            mjd = mjd[1:]
            clk = clk[1:]
        self._time = Time(mjd, format="pulsar_mjd", scale="utc")
        self._clock = clk * u.s

    @staticmethod
    def load_tempo2_clock_file(filename):
        """Read a tempo2-format clock file.

        Returns three values:
        (mjd, clk, hdrline).  The first two are float arrays of MJD and
        clock corrections (seconds).  hdrline is the first line of the file
        that specifies the two clock scales connected by the file.
        """
        with open_or_use(filename, "r") as f:
            hdrline = f.readline().rstrip()
            try:
                mjd, clk = np.loadtxt(f, usecols=(0, 1), unpack=True)
            except (FileNotFoundError, ValueError):
                log.error("Failed loading clock file {0}".format(f))
                raise
        return mjd, clk, hdrline


def write_tempo2_clock_file(filename, hdrline, clock, comments=None):
    """Write clock corrections as a TEMPO2-format clock correction file.

    Parameters
    ----------
    filename : str or pathlib.Path or file-like
        The destination
    hdrline : str
        The first line of the file. Should start with `#` and consist
        of a pair of timescales, like `UTC(AO) UTC(GPS)` that this clock
        file transforms between.
    clock : ClockFile
        ClockFile object to write out.
    comments : str
        Additional comments to include. Lines should probably start with `#`
        so they will be interpreted as comments. This field frequently
        contains details of the origin of the file, or at least the
        commands used to convert it from its original format.
    """
    if not hdrline.startswith("#"):
        raise ValueError(f"Header line must start with #: {hdrline!r}")
    mjds = clock.time.mjd
    corr = clock.clock
    # TEMPO2 writes seconds
    a = np.array([mjds, corr.to_value(u.s)]).T
    header = hdrline.strip() + "\n"
    if comments is not None:
        header += comments
    np.savetxt(filename, a, header=header)


class TempoClockFile(ClockFile):

    format = "tempo"

    def __init__(self, filename, obscode=None, bogus_last_correction=False, **kwargs):
        self.filename = filename
        self.obscode = obscode
        log.debug(
            f"Loading {self.format} observatory ({obscode}) clock correction file {filename}"
        )
        try:
            mjd, clk = self.load_tempo1_clock_file(filename, site=obscode)
        except (FileNotFoundError, OSError):
            log.error(
                f"TEMPO-style clock correction file {filename} for site {obscode} not found"
            )
            mjd = np.array([], dtype=float)
            clk = np.array([], dtype=float)
        if bogus_last_correction and len(mjd):
            mjd = mjd[:-1]
            clk = clk[:-1]
        while len(mjd) and mjd[0] == 0:
            # Zap leading zeros
            mjd = mjd[1:]
            clk = clk[1:]
        # FIXME: using Time may make loading clock corrections slow
        self._time = Time(mjd, format="pulsar_mjd", scale="utc")
        self._clock = clk * u.us

    @staticmethod
    def load_tempo1_clock_file(filename, site=None):
        """Load a TEMPO format clock file for a site

        Given the specified full path to the tempo1-format clock file,
        will return two numpy arrays containing the MJDs and the clock
        corrections (us).  All computations here are done as in tempo, with
        the exception of the 'F' flag (to disable interpolation), which
        is currently not implemented.

        INCLUDE statments are processed.

        If the 'site' argument is set to an appropriate one-character tempo
        site code, only values for that site will be returned. If the 'site'
        argument is None, the file is assumed to contain only clock corrections
        for the desired telescope, so all values found in the file will be returned
        but INCLUDEs will *not* be processed.
        """
        # TODO we might want to handle 'f' flags by inserting addtional
        # entries so that interpolation routines will give the right result.
        # The way TEMPO interprets 'f' flags is that an MJD with an 'f' flag
        # gives the constant clock correction value for at most a day either side.
        # If the sought-after clock correction is within a day of two different
        # 'f' values, the nearest is used.
        # https://github.com/nanograv/tempo/blob/618afb2e901d3e4b8324d4ba12777c055e128696/src/clockcor.f#L79
        # This could be (roughly) implemented by splicing in additional clock correction points
        # between 'f' values or +- 1 day.
        # (The deviation would be that in gaps you get an interpolated value rather than
        # an error message.)
        mjds = []
        clkcorrs = []
        for l in lines_of(filename):
            # Ignore comment lines
            if l.startswith("#"):
                continue

            # Process INCLUDE
            # Assumes included file is in same dir as this one
            if l.startswith("INCLUDE"):
                if site is not None:
                    clkdir = os.path.dirname(os.path.abspath(filename))
                    filename1 = os.path.join(clkdir, l.split()[1])
                    mjds1, clkcorrs1 = TempoClockFile.load_tempo1_clock_file(
                        filename1, site=site
                    )
                    mjds.extend(mjds1)
                    clkcorrs.extend(clkcorrs1)
                continue

            # Parse MJD
            try:
                mjd = float(l[0:9])
                # allow mjd=0 to pass, since that is often used
                # for effectively null clock files
                if (mjd < 39000 and mjd != 0) or mjd > 100000:
                    mjd = None
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
            if (site is not None) and (site.lower() != csite):
                continue

            # Need MJD and at least one of the two clkcorrs
            if mjd is None:
                continue
            if (clkcorr1 is None) and (clkcorr2 is None):
                continue
            # If one of the clkcorrs is missing, it defaults to zero
            if clkcorr1 is None:
                clkcorr1 = 0.0
            if clkcorr2 is None:
                clkcorr2 = 0.0
            # This adjustment is hard-coded in tempo:
            if clkcorr1 > 800.0:
                clkcorr1 -= 818.8
            # Add the value to the list
            mjds.append(mjd)
            clkcorrs.append(clkcorr2 - clkcorr1)

        return mjds, clkcorrs


def write_tempo_clock_file(filename, obscode, clock, comments=None):
    """Write clock corrections as a TEMPO-format clock correction file.

    Parameters
    ----------
    filename : str or pathlib.Path or file-like
        The destination
    obscode : str
        The TEMPO observatory code. TEMPO effectively concatenates
        all its clock corrections and uses this field to determine
        which observatory the clock corrections are relevant to.
        TEMPO observatory codes are case-insensitive one-character values agreed upon
        by convention and occurring in tim files. PINT recognizes
        these as aliases of observatories for which they are agreed upon,
        and an Observatory object contains a field that can be used to
        retrieve this.
    clock : ClockFile
        ClockFile object to write out.
    comments : str
        Additional comments to include. These will be included below the headings
        and each line should be prefaced with `#` to avoid conflicting with the
        clock corrections.
    """
    if not isinstance(obscode, str) or len(obscode) != 1:
        raise ValueError(
            "Invalid TEMPO obscode {obscode!r}, should be one printable character"
        )
    mjds = clock.time.mjd
    corr = clock.clock
    # TEMPO writes microseconds
    a = np.array([mjds, corr.to_value(u.us)]).T
    with open_or_use(filename) as f:
        f.write(
            dedent(
                """\
               MJD       EECO-REF    NIST-REF NS      DATE    COMMENTS
            =========    ========    ======== ==    ========  ========
            """
            )
        )
        if comments is not None:
            f.write(comments.strip())
            f.write("\n")
        # Do not use EECO-REF column as TEMPO does a weird subtraction thing
        for mjd, corr in a:
            # 0:9 for MJD
            # 9:21 for clkcorr1 (do not use)
            # 21:33 for clkcorr2
            # 34 for obscode
            # Extra stuff ignored
            # FIXME: always use C locale
            date = Time(mjd, format="pulsar_mjd").datetime.strftime("%d-%b-%y")
            eeco = 0.0
            f.write(f"{mjd:8.2f}{eeco:12.1}{corr:12.3f} {obscode}    {date}\n")
