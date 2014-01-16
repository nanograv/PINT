"""Miscellaneous potentially-helpful functions.

"""
import math
import numpy as np
import string
from warnings import warn
import mpmath
import astropy.time
import astropy.units as u

class PosVel(object):
    """Position/Velocity class.

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
            self.pos = np.asarray(pos)

        if len(vel) != 3:
            raise ValueError(
                "Position vector has length %d instead of 3" % len(pos))
        if isinstance(vel, u.Quantity):
            self.vel = vel
        else:
            self.vel = np.asarray(vel)

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
    """Convert Fortran-format floating-point strings.

    returns a copy of the input string with all 'D' or 'd' turned
    into 'e' characters.  Intended for dealing with exponential
    notation in tempo1-generated parfiles.
    """
    return float(x.translate(string.maketrans('Dd', 'ee')))


def time_from_mjd_string(s, scale='utc'):
    """Returns an astropy Time object generated from a MJD string input.
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
    """Print an MJD time with lots of digits (number is 'prec').

    astropy does not seem to provide this capability (yet?).
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
    """Return an astropy Time value as MJD in mpmath float format.
    
    mpmath.mp.dps needs to be set to the desired precision before
    calling this.
    """
    return mpmath.mpf(t.jd1 - astropy.time.core.MJD_ZERO) \
            + mpmath.mpf(t.jd2)


def timedelta_to_mpf_sec(t):
    """Return astropy TimeDelta as mpmath value in seconds.
    """
    return (mpmath.mpf(t.jd1)
            + mpmath.mpf(t.jd2))*astropy.time.core.SECS_PER_DAY


def GEO_WGS84_to_ITRF(lon, lat, hgt):
    """Convert lat/long/height to rectangular.

    Convert WGS-84 references lon, lat, height (using astropy
    units) to ITRF x,y,z rectangular coords (m)
    .
    """
    x, y, z = astropy.time.erfa_time.era_gd2gc(1, lon.to(u.rad).value,
                                               lat.to(u.rad).value,
                                               hgt.to(u.m).value)
    return x * u.m, y * u.m, z * u.m



def numeric_partial(f, args, ix=0, delta=1e-6):
    """Compute the partial derivative of f numerically.

    This uses symmetric differences to estimate the partial derivative
    of a function (that takes some number of numeric arguments and may
    return an array) with respect to one of its arguments.
    """
    #r = np.array(f(*args))
    args2 = list(args)
    args2[ix] = args[ix]+delta/2.
    r2 = np.array(f(*args2))
    args3 = list(args)
    args3[ix] = args[ix]-delta/2.
    r3 = np.array(f(*args3))
    return (r2-r3)/delta

def numeric_partials(f, args, delta=1e-6):
    """Compute all the partial derivatives of f numerically.

    Returns a matrix of the partial derivative of every return value
    with respect to every input argument. f is assumed to take a flat list
    of numeric arguments and return a list or array of values.
    """
    r = [numeric_partial(f, args, i, delta) for i in range(len(args))]
    return np.array(r).T

def check_all_partials(f, args, delta=1e-6, atol=1e-4, rtol=1e-4):
    """Check the partial derivatives of a function that returns derivatives.

    The function is assumed to return a pair (values, partials), where
    partials is supposed to be a matrix of the partial derivatives of f
    with respect to all its arguments. These values are checked against
    numerical partial derivatives.
    """
    _, jac = f(*args)
    jac = np.asarray(jac)
    njac = numeric_partials(lambda *args: f(*args)[0], args, delta)

    try:
        np.testing.assert_allclose(jac, njac, atol=atol, rtol=rtol)
    except AssertionError:
        #print jac
        #print njac
        d = np.abs(jac-njac)/(atol+rtol*np.abs(njac))
        print "fail fraction:", np.sum(d > 1)/float(np.sum(d >= 0))
        worst_ix = np.unravel_index(np.argmax(d.reshape((-1,))), d.shape)
        print "max fail:", np.amax(d), "at", worst_ix
        print "jac there:", jac[worst_ix], "njac there:", njac[worst_ix]
        raise
