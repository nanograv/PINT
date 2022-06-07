"""Routines for reading various formats of clock file."""

import os
import re
from pathlib import Path
from textwrap import dedent

import astropy.units as u
import numpy as np
from loguru import logger as log

from pint.pulsar_mjd import Time
from pint.utils import lines_of, open_or_use
from pint.observatory.global_clock_corrections import get_clock_correction_file
from pint.pulsar_mjd import Time


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
        super().__init__(name, bases, members)


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

    def __init__(self):
        # FIXME: require filename?
        self._time = None
        self._clock = None
        self.comments = None

    @classmethod
    def read(cls, filename, format="tempo", **kwargs):
        if format in cls._formats:
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

        If values past the end are encountered, check for new clock corrections
        in the global repository. Delegates the actual computation to the
        included ClockFile object; anything still not covered is treated
        according to ``limits``.

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

        This combines the clock correction files to produce a single file that
        produces clock corrections that are the *sum* of the original input files.
        For example, one could combine `ao2gps.clk` and `gps2utc.clk` to produce
        `ao2utc.clk`.

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
            il = np.searchsorted(times.mjd, b)
            ir = np.searchsorted(times.mjd, e, side="right")
            times = times[il:ir]
            corr = corr[il:ir]

        comments = []
        indices = [0] * len(clocks)
        for m in times.mjd:
            com = ""
            for i, c in enumerate(clocks):
                while indices[i] < len(c._time) and c._time.mjd[indices[i]] < m:
                    indices[i] += 1
                if indices[i] < len(c._time) and c._time.mjd[indices[i]] == m:
                    if com == "":
                        com = c.comments[indices[i]]
                    elif c.comments[indices[i]] != "":
                        com += "\n# " + c.comments[indices[i]]
                    # bump up by 1 in case this is a repeated MJD
                    indices[i] += 1
            comments.append(com)
        r = ConstructedClockFile(
            mjd=times.mjd,
            clock=corr,
            comments=comments,
            filename=f"Merged from {[c.filename for c in clocks]}",
        )

        leading_comment = None
        for c in clocks:
            if c.leading_comment is not None:
                if leading_comment is None:
                    leading_comment = c.leading_comment
                else:
                    leading_comment += "\n" + c.leading_comment
        r.leading_comment = leading_comment

        return r

    def write_tempo_clock_file(self, filename, obscode, extra_comment=None):
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
        comments : str
            Additional comments to include. These will be included below the headings
            and each line should be prefaced with `#` to avoid conflicting with the
            clock corrections.
        """
        if not isinstance(obscode, str) or len(obscode) != 1:
            raise ValueError(
                "Invalid TEMPO obscode {obscode!r}, should be one printable character"
            )
        mjds = self.time.mjd
        corr = self.clock.to_value(u.us)
        comments = self.comments if self.comments else [""] * len(self.clock)
        # TEMPO writes microseconds
        if extra_comment is not None:
            if self.leading_comment is not None:
                leading_comment = extra_comment.rstrip() + "\n" + self.leading_comment
            else:
                leading_comment = extra_comment.rstrip()
        else:
            leading_comment = self.leading_comment
        with open_or_use(filename, "wt") as f:
            f.write(tempo_standard_header)
            if leading_comment is not None:
                f.write(leading_comment.strip())
                f.write("\n")
            # Do not use EECO-REF column as TEMPO does a weird subtraction thing
            for mjd, corr, comment in zip(mjds, corr, comments):
                # 0:9 for MJD
                # 9:21 for clkcorr1 (do not use)
                # 21:33 for clkcorr2
                # 34 for obscode
                # Extra stuff ignored
                # FIXME: always use C locale
                date = Time(mjd, format="pulsar_mjd").datetime.strftime("%d-%b-%y")
                eeco = 0.0
                f.write(f"{mjd:9.2f}{eeco:12.3f}{corr:12.3f} {obscode}    {date}")
                if comment:
                    # Try to avoid trailing whitespace
                    if comment.startswith("\n"):
                        f.write(f"{comment}".rstrip())
                    else:
                        f.write(f"  {comment}".rstrip())
                f.write("\n")

    def write_tempo2_clock_file(self, filename, hdrline=None, extra_comment=None):
        """Write clock corrections as a TEMPO2-format clock correction file.

        Parameters
        ----------
        filename : str or pathlib.Path or file-like
            The destination
        hdrline : str
            The first line of the file. Should start with `#` and consist
            of a pair of timescales, like `UTC(AO) UTC(GPS)` that this clock
            file transforms between.
        comments : str
            Additional comments to include. Lines should probably start with `#`
            so they will be interpreted as comments. This field frequently
            contains details of the origin of the file, or at least the
            commands used to convert it from its original format.
        """
        # Tempo2 requires headerlines to look like "# CLK1 CLK2"
        # listing two time scales
        # Tempo2 can't cope with lines longer than 1023 characters
        # Initial lines starting with # are ignored (not indented)
        # https://bitbucket.org/psrsoft/tempo2/src/master/tabulatedfunction.C
        # Lines starting with # (not indented) are skipped
        # `sscanf("%lf %lf")` means anything after the two values is ignored
        # Also any line that doesn't start with two floats is ignored
        # decreasing forbidden
        if hdrline is None:
            # Assume this was a TEMPO2-format file and use the header
            hdrline = self.header
        if not hdrline.startswith("#"):
            raise ValueError(f"Header line must start with #: {hdrline!r}")
        if extra_comment is not None:
            if self.leading_comment is not None:
                leading_comment = extra_comment.rstrip() + "\n" + self.leading_comment
            else:
                leading_comment = extra_comment.rstrip()
        else:
            leading_comment = self.leading_comment
        with open_or_use(filename, "wt") as f:
            f.write(hdrline.rstrip())
            f.write("\n")
            if leading_comment is not None:
                f.write(leading_comment.rstrip())
                f.write("\n")
            comments = self.comments if self.comments else [""] * len(self.time)

            for mjd, corr, comment in zip(
                self.time.mjd, self.clock.to_value(u.s), comments
            ):
                f.write(f"{mjd:.5f} {corr:.12f}")
                if comment:
                    if not comment.startswith("\n"):
                        f.write(" ")
                    f.write(comment.rstrip())
                f.write("\n")


class ConstructedClockFile(ClockFile):

    # No format set because these can't be read

    def __init__(
        self, mjd, clock, comments=None, leading_comment=None, filename=None, **kwargs
    ):
        super().__init__()
        if len(mjd) != len(clock):
            raise ValueError(f"MJDs have {len(mjd)} entries but clock has {len(clock)}")
        self._time = Time(mjd, format="pulsar_mjd", scale="utc")
        self._clock = clock.to(u.us)
        self.filename = "Constructed" if filename is None else filename
        if comments is None:
            self.comments = [""] * len(self._time)
        else:
            self.comments = comments
            if len(comments) != len(mjd):
                raise ValueError("Comments list does not match time array")
        self.leading_comment = None


class Tempo2ClockFile(ClockFile):

    format = "tempo2"

    hdrline_re = re.compile(r"#\s*(\S+)\s+(\S+)\s+(\d+)?(.*)")
    # This horror is based on https://docs.python.org/3/library/re.html#simulating-scanf
    clkcorr_re = re.compile(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)"
        r"\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)"
        r" ?(.*)"
    )

    def __init__(self, filename, bogus_last_correction=False, **kwargs):
        super().__init__()
        self.filename = filename
        log.debug(f"Loading {self.format} observatory clock correction file {filename}")
        try:
            mjd = []
            clk = []
            self.leading_comment = None
            self.comments = []

            def add_comment(s):
                if self.comments:
                    if self.comments[-1] is None:
                        self.comments[-1] = s.rstrip()
                    else:
                        self.comments[-1] += "\n" + s.rstrip()
                elif self.leading_comment is None:
                    self.leading_comment = s.rstrip()
                else:
                    self.leading_comment += "\n" + s.rstrip()

            with open_or_use(filename) as f:
                hdrline = None
                for line in f:
                    if hdrline is None:
                        hdrline = line
                        m = self.hdrline_re.match(hdrline)
                        if not m:
                            raise ValueError(
                                f"Header line must start with # and contain two time scales: {hdrline!r}"
                            )
                        self.header = hdrline
                        self.timescale_from = m.group(1)
                        self.timescale_to = m.group(2)
                        self.badness = 1 if m.group(3) is None else int(m.group(3))
                        # Extra stuff on the hdrline <shrug />
                        self.hdrline_extra = "" if m.group(4) is None else m.group(4)
                        continue
                    if line.startswith("#"):
                        add_comment(line)
                        continue
                    m = self.clkcorr_re.match(line)
                    if m is None:
                        # Anything that doesn't match is a comment, what fun!
                        # This is what T2 does, using sscanf
                        add_comment(line)
                        continue
                    mjd.append(float(m.group(1)))
                    clk.append(float(m.group(2)))
                    self.comments.append(None)
                    if m.group(3) is not None:
                        # Anything else on the line is a comment too
                        add_comment(m.group(3))
            clk = np.array(clk)
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


tempo_standard_header = dedent(
    """\
       MJD       EECO-REF    NIST-REF NS      DATE    COMMENTS
    =========    ========    ======== ==    ========  ========
    """
)
tempo_standard_header_res = [
    re.compile(
        r"\s*MJD\s+EECO-REF\s+NIST-REF\s+NS\s+DATE\s+COMMENTS\s*", flags=re.IGNORECASE
    ),
    re.compile(r"\s*=+\s+=+\s+=+\s+=+\s+=+\s+=+\s*"),
]


class TempoClockFile(ClockFile):
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

    format = "tempo"

    def __init__(
        self,
        filename,
        obscode=None,
        bogus_last_correction=False,
        process_includes=True,
        **kwargs,
    ):
        super().__init__()
        self.filename = filename
        self.obscode = obscode
        self.leading_comment = None
        log.debug(
            f"Loading {self.format} observatory ({obscode}) clock correction file {filename}"
        )
        mjds = []
        clkcorrs = []
        self.comments = []
        seen_obscodes = set()

        def add_comment(s):
            if self.comments:
                if self.comments[-1] is None:
                    self.comments[-1] = s.rstrip()
                else:
                    self.comments[-1] += "\n" + s.rstrip()
            elif self.leading_comment is None:
                self.leading_comment = s.rstrip()
            else:
                self.leading_comment += "\n" + s.rstrip()

        try:
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
            seen_header = 0

            for l in lines_of(filename):
                # Ignore comment lines
                if l.startswith("#"):
                    add_comment(l)
                    continue

                # TEMPO has very, ah, flexible notions of what is an acceptable file
                # https://sourceforge.net/p/tempo/tempo/ci/master/tree/src/newsrc.f#l272
                # Any line that starts with "MJD" or "=====" is assumed to be part of the header.
                # TEMPO describes this as a "commonly used header format".
                ls = l.split()
                if ls and ls[0].upper().startswith("MJD"):
                    # Header line. Do we preserve it?
                    continue
                if ls and ls[0].startswith("====="):
                    # Header line. Do we preserve it?
                    continue

                # Process INCLUDE
                # Assumes included file is in same dir as this one
                ls = l.split()
                if ls and ls[0].upper() == "INCLUDE" and process_includes:
                    # Find the new file, if possible
                    if isinstance(filename, str):
                        fn = Path(filename)
                    elif isinstance(filename, Path):
                        fn = filename
                    else:
                        raise ValueError(
                            f"Don't know how to process INCLUDE statement in {filename}"
                        )
                    # Construct a TEMPO-format clock file object
                    ifn = fn.parent / ls[1]
                    ic = TempoClockFile(ifn, obscode=obscode)
                    # Splice in that object, handling leading and in-line comments
                    if self.leading_comment is None:
                        if ic.leading_comment is not None:
                            self.leading_comment = ic.leading_comment
                    else:
                        self.leading_comment += "\n" + ic.leading_comment
                    mjds.extend(ic._time.mjds)
                    clkcorrs.extend(ic._clock)
                    self.comments.extend(ic.comments)

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
                if (obscode is not None) and (obscode.lower() != csite):
                    continue
                # FIXME: f flag(?) in l[36]?
                if csite is not None:
                    seen_obscodes.add(csite)
                    if len(seen_obscodes) > 1:
                        raise ValueError(
                            f"TEMPO-format file {filename} contains multiple "
                            f"observatory codes: {seen_obscodes}"
                        )

                # Need MJD and at least one of the two clkcorrs
                if mjd is None:
                    add_comment(l)
                    continue
                if (clkcorr1 is None) and (clkcorr2 is None):
                    add_comment(l)
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
                self.comments.append(None)
                add_comment(l[50:])
        except (FileNotFoundError, OSError):
            log.error(
                f"TEMPO-style clock correction file {filename} for site {obscode} not found"
            )
        if bogus_last_correction and len(mjds):
            mjds = mjds[:-1]
            clkcorrs = clkcorrs[:-1]
            # FIXME: do something sensible with comments!
        while len(mjds) and mjds[0] == 0:
            # Zap leading zeros
            mjds = mjds[1:]
            clkcorrs = clkcorrs[1:]
            # FIXME: do something sensible with comments!
        # FIXME: using Time may make loading clock corrections slow
        self._time = Time(mjds, format="pulsar_mjd", scale="utc")
        self._clock = clkcorrs * u.us


class GlobalClockFile(ClockFile):
    """Clock file obtained from a global repository.

    These clock files are downloaded from a global repository; if a TOA
    is encountered past the end of the current version, the code will
    reach out to the global repository looking for a new version.

    This supports both TEMPO- and TEMPO2-format files; just instantiate the
    object with appropriate arguments and it will call
    :func:`pint.observatory.ClockFile.read` with the right arguments.
    """

    # FIXME: fall back to built-in?

    def __init__(self, filename, format="tempo", **kwargs):
        self.filename = filename
        self.format = format
        self.kwargs = kwargs
        f = get_clock_correction_file(self.filename, download_policy="if_missing")
        self.clock_file = ClockFile.read(f, self.format, **kwargs)

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
        if np.any(t > self.clock_file.time[-1]):
            f = get_clock_correction_file(self.filename)
            self.clock_file = ClockFile.read(f, format=self.format, **self.kwargs)
        return self.clock_file.evaluate(t, limits=limits)

    @property
    def leading_comment(self):
        return self.clock_file.leading_comment

    @property
    def comments(self):
        return self.clock_file.comments

    @property
    def time(self):
        return self.clock_file.time

    @property
    def clock(self):
        return self.clock_file.clock

    def last_clock_correction_mjd(self):
        return self.clock_file.last_clock_correction_mjd()
