"""Routines for reading and writing various formats of clock file."""

import re
import warnings
from pathlib import Path
from textwrap import dedent
from warnings import warn

import astropy.units as u
import erfa
import numpy as np
from loguru import logger as log

from pint.observatory import ClockCorrectionOutOfRange, NoClockCorrections
from pint.observatory.global_clock_corrections import get_clock_correction_file
from pint.pulsar_mjd import Time
from pint.utils import compute_hash, lines_of, open_or_use

__all__ = [
    "ClockFile",
    "GlobalClockFile",
]


class ClockFile:
    """A clock correction file.

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

    Clock correction file objects preserve the comments in the original file,
    and can be written in TEMPO or TEMPO2 format.

    If you want to provide a new clock correction file format, you should
    write a function that can read it and return an appropriate ClockFile
    object. It should take a filename, Path, or file-like object as its
    first argument, and probably at least a ``friendly_name`` keyword argument.
    You would then add it to ``ClockFile._formats`` as something like
    ``ClockFile._formats["haiku"] = my_read_haiku_function``. Then
    ``ClockFile.read(filename, format="haiku", ...)`` will call
    ``my_read_haiku_function(filename, ...)``.

    Parameters
    ----------
    mjd : np.ndarray
        The MJDs at which clock corrections are measured
    clock : astropy.units.Quantity
        The clock corrections at those MJDs (units of time)
    comments : list of str or None
        The comments following each clock correction; should match ``mjd``
        and ``clock`` in length. If not provided, a list of empty comments
        is used. The first line of these strings are normally on the same
        line as the corresponding clock correction, while any subsequent
        lines occur between it and the next and should probably start with
        ``# ``.
    leading_comment : str
        A comment to put at the top of the file. Lines should probably start
        with ``# ``.
    filename : str or None
        If present, a file that can be read to reproduce this data.
    friendly_name : str or None
        A descriptive file name, in case the filename is long or uninformative.
        If not provided defaults to the filename.
    header : str
        A header to include, if output in TEMPO2 format.
    """

    _formats = {}

    def __init__(
        self,
        mjd,
        clock,
        *,
        filename=None,
        friendly_name=None,
        comments=None,
        header=None,
        leading_comment=None,
        valid_beyond_ends=False,
    ):
        self.filename = filename
        self.friendly_name = self.filename if friendly_name is None else friendly_name
        self.valid_beyond_ends = valid_beyond_ends
        if len(mjd) != len(clock):
            raise ValueError(f"MJDs have {len(mjd)} entries but clock has {len(clock)}")
        self._time = Time(mjd, format="pulsar_mjd", scale="utc")
        if not np.all(np.diff(self._time.mjd) >= 0):
            i = np.where(np.diff(self._time.mjd) < 0)[0][0]
            raise ValueError(
                f"Clock file {self.friendly_name} appears to be out of order: {self._time[i]} > {self._time[i+1]}"
            )
        self._clock = clock.to(u.us)
        if comments is None:
            self.comments = [""] * len(self._time)
        else:
            self.comments = comments
            if len(comments) != len(mjd):
                raise ValueError("Comments list does not match time array")
        self.leading_comment = "" if leading_comment is None else leading_comment
        self.header = header

    @classmethod
    def read(cls, filename, format="tempo", **kwargs):
        """Read file, selecting an appropriate subclass based on format.

        Any additional keyword arguments are passed to the appropriate reader function.
        """
        if format in cls._formats:
            return cls._formats[format](filename, **kwargs)
        else:
            raise ValueError(f"clock file format '{format}' not defined")

    @property
    def time(self):
        """An astropy.time.Time recording the dates of clock corrections."""
        return self._time

    @property
    def clock(self):
        """An astropy.units.Quantity recording the amounts of clock correction."""
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
        if not self.valid_beyond_ends and len(self.time) == 0:
            msg = f"No data points in clock file '{self.friendly_name}'"
            if limits == "warn":
                warn(msg)
                return np.zeros_like(t) * u.us
            elif limits == "error":
                raise NoClockCorrections(msg)

        if not self.valid_beyond_ends and (
            np.any(t < self.time[0]) or np.any(t > self.time[-1])
        ):
            msg = f"Data points out of range in clock file '{self.friendly_name}'"
            if limits == "warn":
                warn(msg)
            elif limits == "error":
                raise ClockCorrectionOutOfRange(msg)

        # Can't pass Times directly to np.interp.  This should be OK:
        return np.interp(t.mjd, self.time.mjd, self.clock.to(u.us).value) * u.us

    def last_correction_mjd(self):
        """Last MJD for which corrections are available."""
        return -np.inf if len(self.time) == 0 else self.time[-1].mjd

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
            b = max(c._time.mjd[0] for c in clocks)
            e = min(c._time.mjd[-1] for c in clocks)
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

        leading_comment = f"# Merged from {[c.friendly_name for c in clocks]}"
        for c in clocks:
            if c.leading_comment is not None:
                if leading_comment is None:
                    leading_comment = c.leading_comment
                else:
                    leading_comment += "\n" + c.leading_comment

        return ClockFile(
            mjd=times.mjd,
            clock=corr,
            comments=comments,
            leading_comment=leading_comment,
            friendly_name=f"Merged from {[c.filename for c in clocks]}",
        )

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
        comments = self.comments or [""] * len(self.clock)
        # TEMPO writes microseconds
        if extra_comment is None:
            leading_comment = self.leading_comment
        elif self.leading_comment is not None:
            leading_comment = extra_comment.rstrip() + "\n" + self.leading_comment
        else:
            leading_comment = extra_comment.rstrip()
        with open_or_use(filename, "wt") as f:
            f.write(tempo_standard_header)
            if leading_comment is not None:
                f.write(leading_comment.strip())
                f.write("\n")
            # Do not use EECO-REF column as TEMPO does a weird subtraction thing
            # sourcery skip: hoist-statement-from-loop
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
            file transforms between. If no value is provided, the value of
            ``self.header`` is used.
        extra_comment : str
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
        if extra_comment is None:
            leading_comment = self.leading_comment
        elif self.leading_comment is not None:
            leading_comment = extra_comment.rstrip() + "\n" + self.leading_comment
        else:
            leading_comment = extra_comment.rstrip()
        with open_or_use(filename, "wt") as f:
            f.write(hdrline.rstrip())
            f.write("\n")
            if leading_comment is not None:
                f.write(leading_comment.rstrip())
                f.write("\n")
            comments = self.comments or [""] * len(self.time)

            for mjd, corr, comment in zip(
                self.time.mjd, self.clock.to_value(u.s), comments
            ):
                f.write(f"{mjd:.5f} {corr:.12f}")
                if comment:
                    if not comment.startswith("\n"):
                        f.write(" ")
                    f.write(comment.rstrip())
                f.write("\n")

    def export(self, filename):
        """Write this clock correction file to the specified location."""
        # FIXME: fall back to writing the clock file using .write_...?
        if self.filename is None:
            raise ValueError("No file backing this clock correction object")
        try:
            contents = Path(self.filename).read_text()
        except IOError:
            if len(self.time) > 0:
                log.info(f"Unable to load original clock file for {self}")
                # FIXME: use write? Do we know what format we should be in?
            return
        with open_or_use(filename, "wt") as f:
            f.write(contents)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.friendly_name=}, {len(self.time)=})"


# TEMPO2

hdrline_re = re.compile(r"#\s*(\S+)\s+(\S+)\s+(\d+)?(.*)")
# This horror is based on https://docs.python.org/3/library/re.html#simulating-scanf
clkcorr_re = re.compile(
    r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)"
    r"\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)"
    r" ?(.*)"
)


def read_tempo2_clock_file(
    filename, bogus_last_correction=False, friendly_name=None, valid_beyond_ends=False
):
    """Read a TEMPO2-format clock file.

    This function can also be accessed through
    :func:`pint.observatory.clock_file.ClockFile.read` with the
    ``format="tempo2"`` argument.

    Parameters
    ----------
    filename : str or pathlib.Path or file-like
        The location to obtain the file from.
    bogus_last_correction : bool
        If True, the last correction value in the file is a placeholder, normally
        far in the future, and not an actual measurement.
    friendly_name : str or None
        A human-readable name for this file, for use in error reporting.
        If not provided, will default to ``filename``.
    valid_beyond_ends : bool
        Whether to consider the file valid past the ends of the data it contains.
    """
    log.debug(
        f"Loading TEMPO2-format observatory clock correction file {friendly_name} ({filename}) with {bogus_last_correction=}"
    )
    try:
        mjd = []
        clk = []
        leading_comment = None
        comments = []

        def add_comment(s):
            nonlocal leading_comment
            if comments:
                if comments[-1] is None:
                    comments[-1] = s.rstrip()
                else:
                    comments[-1] += "\n" + s.rstrip()
            elif leading_comment is None:
                leading_comment = s.rstrip()
            else:
                leading_comment += "\n" + s.rstrip()

        with open_or_use(filename) as f:
            hdrline = None
            for line in f:
                if hdrline is None:
                    hdrline = line
                    m = hdrline_re.match(hdrline)
                    if not m:
                        raise ValueError(
                            f"Header line must start with # and contain two time scales: {hdrline!r}"
                        )
                    header = hdrline
                    # FIXME: explicit support of from/to timescales and badness?
                    timescale_from = m.group(1)
                    timescale_to = m.group(2)
                    badness = 1 if m.group(3) is None else int(m.group(3))
                    # Extra stuff on the hdrline <shrug />
                    hdrline_extra = "" if m.group(4) is None else m.group(4)
                    continue
                if line.startswith("#"):
                    add_comment(line)
                    continue
                m = clkcorr_re.match(line)
                if m is None:
                    # Anything that doesn't match is a comment, what fun!
                    # This is what T2 does, using sscanf
                    add_comment(line)
                    continue
                mjd.append(float(m.group(1)))
                clk.append(float(m.group(2)))
                comments.append(None)
                if m.group(3) is not None:
                    # Anything else on the line is a comment too
                    add_comment(m.group(3))
        clk = np.array(clk)
    except OSError:
        raise NoClockCorrections(
            f"TEMPO2-style clock correction file {filename} not found"
        )
    if bogus_last_correction and len(mjd):
        mjd = mjd[:-1]
        clk = clk[:-1]
        comments = comments[:-1]
    while len(mjd) and mjd[0] == 0:
        # Zap leading zeros
        mjd = mjd[1:]
        clk = clk[1:]
        comments = comments[1:]
    with warnings.catch_warnings():
        # Some clock files have dubious years in them
        # Most are removed by automatically ignoring MJD 0, or with "bogus_last_correction"
        # But Parkes incudes a non-zero correction for MJD 0 so it isn't removed
        # In any case, the user doesn't need a warning about strange years in clock files
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        return ClockFile(
            mjd,
            clk * u.s,
            filename=filename,
            comments=comments,
            leading_comment=leading_comment,
            header=header,
            friendly_name=friendly_name,
            valid_beyond_ends=valid_beyond_ends,
        )


ClockFile._formats["tempo2"] = read_tempo2_clock_file

# FIXME: `NIST-REF` could be replaced by the two timescales in actual use
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


def read_tempo_clock_file(
    filename,
    obscode=None,
    bogus_last_correction=False,
    process_includes=True,
    friendly_name=None,
    valid_beyond_ends=False,
):
    """Read a TEMPO-format clock file.

    This function can also be accessed through
    :func:`pint.observatory.clock_file.ClockFile.read` with the
    ``format="tempo2"`` argument.

    All computations here are done as in tempo, with
    the exception of the 'F' flag (to disable interpolation), which
    is currently not implemented.

    INCLUDE statements are processed.


    Parameters
    ----------
    filename : str or pathlib.Path or file-like
        The location to obtain the file from.
    obscode : str or None
        If the ``obscode`` argument is set to an appropriate one-character tempo
        site code, only values for that site will be returned. If the ``obscode``
        argument is None, the file is assumed to contain only clock corrections
        for the desired telescope, so all values found in the file will be returned
        but INCLUDEs will *not* be processed.
    bogus_last_correction : bool
        If True, the last correction value in the file is a placeholder, normally
        far in the future, and not an actual measurement.
    process_includes : bool
        If True, also read data from any INCLUDE statements. Requires the ``obscode``
        argument to distinguish data from any other telescopes that might be in the
        same system of files.
    friendly_name : str or None
        A human-readable name for this file, for use in error reporting.
        If not provided, will default to ``filename``.
    valid_beyond_ends : bool
        Whether to consider the file valid past the ends of the data it contains.
    """

    leading_comment = None
    if obscode is None:
        log.debug(
            f"Loading TEMPO-format observatory clock correction file {friendly_name} ({filename}) with {bogus_last_correction=}"
        )
    else:
        log.debug(
            f"Loading TEMPO-format observatory ({obscode}) clock correction file {friendly_name} ({filename}) with {bogus_last_correction=}"
        )

    mjds = []
    clkcorrs = []
    comments = []
    seen_obscodes = set()

    def add_comment(s):
        nonlocal leading_comment
        if comments:
            if comments[-1] is None:
                comments[-1] = s.rstrip()
            else:
                comments[-1] += "\n" + s.rstrip()
        elif leading_comment is None:
            leading_comment = s.rstrip()
        else:
            leading_comment += "\n" + s.rstrip()

    try:
        # TODO we might want to handle 'f' flags by inserting additional
        # entries so that interpolation routines will give the right result.
        # The way TEMPO interprets 'f' flags is that an MJD with an 'f' flag
        # gives the constant clock correction value for at most a day either side.
        # If the sought-after clock correction is within a day of two different
        # 'f' values, the nearest is used.
        # https://github.com/nanograv/tempo/blob/618afb2e901d3e4b8324d4ba12777c055e128696/src/clockcor.f#L79
        # This could be (roughly) implemented by splicing in additional clock correction points
        # between 'f' values or +- 1 day.
        # (The deviation would be that in gaps you get an interpolated value
        # rather than an error message.)

        for l in lines_of(filename):
            # Ignore comment lines
            if l.startswith("#"):
                add_comment(l)
                continue

            # FIXME: the header *could* contain to and from timescale information
            # TEMPO has very, ah, flexible notions of what is an acceptable file
            # https://sourceforge.net/p/tempo/tempo/ci/master/tree/src/newsrc.f#l272
            # Any line that starts with "MJD" or "=====" is assumed to be
            # part of the header.  TEMPO describes this as a
            # "commonly used header format".
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
                ic = read_tempo_clock_file(ifn, obscode=obscode)
                # Splice in that object, handling leading and in-line comments
                if leading_comment is None:
                    if ic.leading_comment is not None:
                        leading_comment = ic.leading_comment
                else:
                    leading_comment += "\n" + ic.leading_comment
                mjds.extend(ic._time.mjds)
                clkcorrs.extend(ic._clock)
                comments.extend(ic.comments)

            # Parse MJD
            try:
                mjd = float(l[:9])
                # allow mjd=0 to pass, since that is often used
                # for effectively null clock files
                if (mjd < 39000 and mjd != 0) or mjd > 100000:
                    log.info(f"Disregarding suspicious MJD {mjd} in TEMPO clock file")
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
            comments.append(None)
            add_comment(l[50:])
    except OSError:
        raise NoClockCorrections(
            f"TEMPO-style clock correction file {filename} "
            f"for site {obscode} not found"
        )
    if bogus_last_correction and len(mjds):
        mjds = mjds[:-1]
        clkcorrs = clkcorrs[:-1]
        comments = comments[:-1]
    while len(mjds) and mjds[0] == 0:
        # Zap leading zeros
        mjds = mjds[1:]
        clkcorrs = clkcorrs[1:]
        comments = comments[1:]
    return ClockFile(
        mjds,
        clkcorrs * u.us,
        filename=filename,
        comments=comments,
        leading_comment=leading_comment,
        friendly_name=friendly_name,
        valid_beyond_ends=valid_beyond_ends,
    )


ClockFile._formats["tempo"] = read_tempo_clock_file


class GlobalClockFile(ClockFile):
    """Clock file obtained from a global repository.

    These clock files are downloaded from a global repository; if a TOA
    is encountered past the end of the current version, the code will
    reach out to the global repository looking for a new version.

    This supports both TEMPO- and TEMPO2-format files; just instantiate the
    object with appropriate arguments and it will call
    :func:`pint.observatory.ClockFile.read` with the right arguments.

    Parameters
    ----------
    filename : str
        The name of the file in the global repository
    format : str
        The name of the format of the file, probably "tempo" or "tempo2"
    url_base : str or None
        Location of the global repository (useful for testing)
    url_mirrors : list of str or None
        Mirrors of the global repository (useful for testing)
    """

    def __init__(
        self, filename, format="tempo", url_base=None, url_mirrors=None, **kwargs
    ):
        # FIXME: should we use super here?
        # I think it'll break because of all the properties
        self.filename = filename
        self.friendly_name = filename
        self.format = format
        log.debug(f"Global clock file {self.friendly_name} saving {kwargs=}")
        self.kwargs = kwargs
        self.url_base = url_base
        self.url_mirrors = url_mirrors
        f = get_clock_correction_file(
            self.filename,
            download_policy="if_missing",
            url_base=self.url_base,
            url_mirrors=self.url_mirrors,
        )
        self.f = f
        self.hash = compute_hash(f)
        self.clock_file = ClockFile.read(
            f, self.format, friendly_name=self.friendly_name, **kwargs
        )

    def update(self):
        """Download a new version of a clock file if appropriate.

        An update is appropriate if the last-downloaded version is older than
        the update frequency specified in ``index.txt``. This function should not
        be called unless data outside the range available in the already present
        clock file is requested, or if the user explicitly requests a new version.
        """
        # FIXME: allow user to force an update? by passing an appropriate
        # download policy to get_clock_correction_file, presumably
        mtime = self.f.stat().st_mtime
        f = get_clock_correction_file(
            self.filename, url_base=self.url_base, url_mirrors=self.url_mirrors
        )
        if f != self.f or f.stat().st_mtime != mtime:
            self.f = f
            h = compute_hash(f)
            if h != self.hash:
                # Actual new data (probably)!
                self.hash = h
                self.clock_file = ClockFile.read(
                    f,
                    format=self.format,
                    friendly_name=self.friendly_name,
                    **self.kwargs,
                )
                self.clock_file.friendly_name = self.friendly_name

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
        needs_update = np.any(t > self.clock_file.time[-1])
        if needs_update:
            self.update()
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

    # FIXME: do we need last_correction_mjd to try an update?

    def export(self, filename):
        """Write this clock correction file to the specified location."""
        # Only the inner file knows where it is actually stored
        self.clock_file.export(filename)
