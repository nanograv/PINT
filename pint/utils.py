# utils.py
# Miscellaneous potentially-helpful functions
import math
import numpy
import string
from warnings import warn
import mpmath
import astropy.time
import astropy.units as u

class PosVel(object):
    """
    PosVel(pos, vel)

    The class is used to represent the 6 values describing position
    and velocity vectors.  Instances have 'pos' and 'vel' attributes
    that are numpy arrays of floats (and can have attached astropy
    units).  The 'pos' and 'vel' params are 3-vectors of the positions
    and velocities respectively.

    The 'obj' and 'origin' components are strings that can optionally
    be used to specify names for endpoints of the vectors.  If present,
    addition/subtraction will check that vectors are being combined in
    a consistent way.
    """
    def __init__(self, pos, vel, obj=None, origin=None):
        if len(pos) != 3:
            raise ValueError(
                "Position vector has length %d instead of 3" % len(pos))
        if isinstance(pos, u.Quantity):
            self.pos = pos
        else:
            self.pos = numpy.asarray(pos)

        if len(vel) != 3:
            raise ValueError(
                "Position vector has length %d instead of 3" % len(pos))
        if isinstance(vel, u.Quantity):
            self.vel = vel
        else:
            self.vel = numpy.asarray(vel)

        self.obj = obj
        self.origin = origin

    def _has_labels(self):
        return (self.obj is not None) and (self.origin is not None)

    def __neg__(self):
        return PosVel(-self.pos, -self.vel, obj=self.origin, origin=self.obj)

    def __add__(self, other):
        obj = None
        origin = None
        if self._has_labels() and other._has_labels():
            # here we check that the addition "makes sense", ie the endpoint
            # of self is the origin of other (or vice-versa)
            if self.obj == other.origin:
                origin = self.origin
                obj = other.obj
            elif self.origin == other.obj:
                origin = other.origin
                obj = self.obj
            else:
                raise RuntimeError("Attempting to add incompatible vectors: " +
                        "%s->%s + %s->%s" % (self.origin, self.obj,
                            other.origin, other.obj))

        return PosVel(self.pos + other.pos, self.vel + other.vel,
                obj=obj, origin=origin)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __str__(self):
        if self._has_labels():
            return (str(self.pos)+", "+str(self.vel)
                    + " " + self.origin + "->" + self.obj)
        else:
            return str(self.pos)+", "+str(self.vel)

def fortran_float(x):
    """
    fortran_float(x)

    returns a copy of the input string with all 'D' or 'd' turned
    into 'e' characters.  Intended for dealing with exponential
    notation in tempo1-generated parfiles.
    """
    return float(x.translate(string.maketrans('Dd', 'ee')))


def time_from_mjd_string(s, scale='utc'):
    """
    time_from_mjd_string(s, scale='utc')

    Returns an astropy Time object generated from a MJD string input.
    """
    ss = s.lower()
    if "e" in ss or "d" in ss:
        ss = ss.translate(string.maketrans("d", "e"))
        num, expon = ss.split("e")
        expon = int(expon)
        if expon < 0:
            warn("Likely bogus sci notation input in "+
                 "time_from_mjd_string ('%s')!" % s)
            # This could cause a loss of precision...
            # maybe throw an exception instead?
            imjd, fmjd = 0, float(ss)
        else:
            imjd_s, fmjd_s = num.split('.')
            imjd = int(imjd_s + fmjd_s[:expon])
            fmjd = float("0."+fmjd_s[expon:])
    else:
        imjd_s, fmjd_s = ss.split('.')
        imjd = int(imjd_s)
        fmjd = float("0." + fmjd_s)
    return astropy.time.Time(imjd, fmjd, scale=scale, format='mjd',
                             precision=9)


def time_to_mjd_string(t, prec=15):
    """
    time_to_mjd_string(t, prec=15)

    Print an MJD time with lots of digits (number is 'prec').  astropy
    does not seem to provide this capability (yet?).
    """
    jd1 = t.jd1 - astropy.time.core.MJD_ZERO
    imjd = int(jd1)
    fjd1 = jd1 - imjd
    fmjd = t.jd2 + fjd1
    assert math.fabs(fmjd) < 2.0
    if fmjd >= 1.0:
        imjd += 1
        fmjd -= 1.0
    if fmjd < 0.0:
        imjd -= 1
        fmjd += 1.0
    fmt = "%."+"%sf"%prec
    return str(imjd) + (fmt%fmjd)[1:]


def time_to_mjd_mpf(t):
    """
    time_to_mjd_mpf(t)

    Return an astropy Time value as MJD in mpmath float format.
    mpmath.mp.dps needs to be set to the desired precision before
    calling this.
    """
    return mpmath.mpf(t.jd1 - astropy.time.core.MJD_ZERO) \
            + mpmath.mpf(t.jd2)


def timedelta_to_mpf_sec(t):
    """
    timedelta_to_mpf_sec(t):

    Return astropy TimeDelta as mpmath value in seconds.
    """
    return (mpmath.mpf(t.jd1)
            + mpmath.mpf(t.jd2))*astropy.time.core.SECS_PER_DAY


def GEO_WGS84_to_ITRF(lon, lat, hgt):
    """
    GEO_WGS84_to_ITRF(lon, lat, hgt):

    Convert WGS-84 references lon, lat, height (using astropy
    units) to ITRF x,y,z rectangular coords (m)
    .
    """
    x, y, z = astropy.time.erfa_time.era_gd2gc(1, lon.to(u.rad).value,
                                               lat.to(u.rad).value,
                                               hgt.to(u.m).value)
    return x * u.m, y * u.m, z * u.m

