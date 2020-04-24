# clock_file.py

# Routines for reading various formats of clock file.
from __future__ import absolute_import, division, print_function

import os
import warnings

import astropy.units as u
import numpy
from astropy import log
from astropy._erfa import ErfaWarning
from six import add_metaclass

from pint.pulsar_mjd import Time


class ClockFileMeta(type):
    """Metaclass that provides a registry for different clock file formats.
    ClockFile implementations should define a 'format' class member giving
    the name of the format."""

    def __init__(cls, name, bases, members):
        regname = "_formats"
        if not hasattr(cls, regname):
            setattr(cls, regname, {})
        if "format" in members:
            getattr(cls, regname)[cls.format] = cls
        super(ClockFileMeta, cls).__init__(name, bases, members)


@add_metaclass(ClockFileMeta)
class ClockFile(object):
    """The ClockFile class provides a way to read various formats of clock
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
            return cls._formats[format](filename, **kwargs)
        else:
            raise ValueError("clock file format '%s' not defined" % format)

    @property
    def time(self):
        return self._time

    @property
    def clock(self):
        return self._clock

    def evaluate(self, t, limits="warn"):
        """Evaluate the clock corrections at the times t (given as an
        array-valued Time object).  By default, values are linearly
        interpolated but this could be overridden by derived classes
        if needed.  The first/last values will be applied to times outside
        the data range.  If limits=='warn' this will also issue a warning.
        If limits=='error' an exception will be raised."""

        if numpy.any(t < self.time[0]) or numpy.any(t > self.time[-1]):
            msg = "Data points out of range in clock file '%s'" % self.filename
            if limits == "warn":
                log.warning(msg)
            elif limits == "error":
                raise RuntimeError(msg)

        # Can't pass Times directly to numpy.interp.  This should be OK:
        return numpy.interp(t.mjd, self.time.mjd, self.clock.to(u.us).value) * u.us


class Tempo2ClockFile(ClockFile):

    format = "tempo2"

    def __init__(self, filename, **kwargs):
        self.filename = filename
        log.debug(
            "Loading {0} observatory clock correction file {1}".format(
                self.format, filename
            )
        )
        mjd, clk, self.header = self.load_tempo2_clock_file(filename)
        # NOTE Clock correction file has a time far in the future as ending point
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            self._time = Time(mjd, format="pulsar_mjd", scale="utc")
        self._clock = clk * u.s

    @staticmethod
    def load_tempo2_clock_file(filename):
        """Reads a tempo2-format clock file.  Returns three values:
        (mjd, clk, hdrline).  The first two are float arrays of MJD and
        clock corrections (seconds).  hdrline is the first line of the file
        that specifies the two clock scales connected by the file."""
        f = open(filename, "r")
        hdrline = f.readline().rstrip()
        try:
            mjd, clk = numpy.loadtxt(f, usecols=(0, 1), unpack=True)
        except:
            log.error("Failed loading clock file {0}".format(f))
            raise
        return mjd, clk, hdrline


class TempoClockFile(ClockFile):

    format = "tempo"

    def __init__(self, filename, obscode=None, **kwargs):
        self.filename = filename
        self.obscode = obscode
        log.debug(
            "Loading {0} observatory ({1}) clock correction file {2}".format(
                self.format, obscode, filename
            )
        )
        mjd, clk = self.load_tempo1_clock_file(filename, site=obscode)
        # NOTE Clock correction file has a time far in the future as ending point
        # We are swithing off astropy warning only for gps correction.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            try:
                self._time = Time(mjd, format="pulsar_mjd", scale="utc")
            except ValueError:
                log.error(
                    "Filename {0}, site {1}: Bad MJD {2}".format(filename, obscode, mjd)
                )
                raise
        self._clock = clk * u.us

    @staticmethod
    def load_tempo1_clock_file(filename, site=None):
        """
        Given the specified full path to the tempo1-format clock file,
        will return two numpy arrays containing the MJDs and the clock
        corrections (us).  All computations here are done as in tempo, with
        the exception of the 'F' flag (to disable interpolation), which
        is currently not implemented.

        INCLUDE statments are processed.

        If the 'site' argument is set to an appropriate one-character tempo
        site code, only values for that site will be returned, otherwise all
        values found in the file will be returned.
        """
        # TODO we might want to handle 'f' flags by inserting addtional
        # entries so that interpolation routines will give the right result.
        mjds = []
        clkcorrs = []
        for l in open(filename).readlines():
            # Ignore comment lines
            if l.startswith("#"):
                continue

            # Process INCLUDE
            # Assumes included file is in same dir as this one
            if l.startswith("INCLUDE"):
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
                if mjd < 39000 or mjd > 100000:
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
