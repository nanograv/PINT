"""Miscellaneous potentially-helpful functions."""
import numpy as np
from scipy.misc import factorial
import string
import astropy.time
try:
    from astropy.erfa import DJM0
except ImportError:
    from astropy._erfa import DJM0
import astropy.units as u
from astropy import log
from .str2ld import str2ldarr1
import re
try:
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

# Define prefix parameter pattern
pp1 = re.compile(r'([a-zA-Z]\d[a-zA-Z]+)(\d+)') # For the prefix like T2EFAC2
pp2 = re.compile(r'([a-zA-Z]+)(\d+)')  # For the prefix like F12
pp3 = re.compile(r'([a-zA-Z0-9]+_*)(\d+)')  # For the prefix like DMXR1_3

prefixPattern = [pp1, pp2, pp3]


class PosVel(object):
    """Position/Velocity class.

    The class is used to represent the 6 values describing position
    and velocity vectors.  Instances have 'pos' and 'vel' attributes
    that are numpy arrays of floats (and can have attached astropy
    units).  The 'pos' and 'vel' params are 3-vectors of the positions
    and velocities respectively.

    The coordinates are generally assumed to be aligned with ICRF (J2000),
    i.e. they are in an intertial, not earth-rotating frame

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
                                                        other.origin,
                                                        other.obj))

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
    try:
        # First treat it as a string, wih d->e
        return float(x.translate(maketrans('Dd', 'ee')))
    except AttributeError:
        # If that didn't work it may already be a numeric type
        return float(x)


def time_from_mjd_string(s, scale='utc'):
    """Returns an astropy Time object generated from a MJD string input."""
    ss = s.lower()
    if "e" in ss or "d" in ss:
        ss = ss.translate(maketrans("d", "e"))
        num, expon = ss.split("e")
        expon = int(expon)
        if expon < 0:
            log.warn("Likely bogus sci notation input in " +
                     "time_from_mjd_string ('%s')!" % s)
            # This could cause a loss of precision...
            # maybe throw an exception instead?
            imjd, fmjd = 0, float(ss)
        else:
            imjd_s, fmjd_s = num.split('.')
            imjd = int(imjd_s + fmjd_s[:expon])
            fmjd = float("0."+fmjd_s[expon:])
    else:
        mjd_s = ss.split('.')
        # If input was given as an integer, add floating "0"
        if len(mjd_s) == 1:
            mjd_s.append("0")
        imjd_s, fmjd_s = mjd_s
        imjd = int(imjd_s)
        fmjd = float("0." + fmjd_s)

    return astropy.time.Time(imjd, fmjd, scale=scale, format='pulsar_mjd',
                             precision=9)


def time_from_longdouble(t, scale='utc'):
    st = longdouble2string(t)
    return time_from_mjd_string(st, scale)


def time_to_mjd_string(t, prec=15):
    """Print an MJD time with lots of digits (number is 'prec').

    astropy does not seem to provide this capability (yet?).
    """
    jd1 = t.jd1 - DJM0
    imjd = int(jd1)
    fjd1 = jd1 - imjd
    fmjd = t.jd2 + fjd1
    assert np.fabs(fmjd) < 2.0
    if fmjd >= 1.0:
        imjd += 1
        fmjd -= 1.0
    if fmjd < 0.0:
        imjd -= 1
        fmjd += 1.0
    fmt = "%." + "%sf" % prec
    return str(imjd) + (fmt % fmjd)[1:]


def time_to_mjd_string_array(t, prec=15):
    """Print and MJD time array from an astropy time object as array in
       time.
    """
    jd1 = np.array(t.jd1)
    jd2 = np.array(t.jd2)
    jd1 = jd1 - DJM0
    imjd = jd1.astype(int)
    fjd1 = jd1 - imjd
    fmjd = jd2 + fjd1

    assert np.fabs(fmjd).max() < 2.0
    s = []
    for i, f in zip(imjd, fmjd):
        if f >= 1.0:
            i += 1
            f -= 1.0
        if f < 0.0:
            i -= 1
            f += 1.0
        fmt = "%." + "%sf" % prec
        s.append(str(i) + (fmt % f)[1:])
    return s


def time_to_longdouble(t):
    """ Return an astropy Time value as MJD in longdouble

    ## SUGGESTION(paulr): This function is at least partly redundant with
    ## ddouble2ldouble() below...

    ## Also, is it certain that this calculation retains the full precision?

    """
    try:
        return np.longdouble(t.jd1 - DJM0) + np.longdouble(t.jd2)
    except:
        return np.longdouble(t)


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
        print("fail fraction:", np.sum(d > 1)/float(np.sum(d >= 0)))
        worst_ix = np.unravel_index(np.argmax(d.reshape((-1,))), d.shape)
        print("max fail:", np.amax(d), "at", worst_ix)
        print("jac there:", jac[worst_ix], "njac there:", njac[worst_ix])
        raise


def has_astropy_unit(x):
    """
    has_astropy_unit(x):

    Return True/False if x has an astropy unit type associated with it. This is
    useful, because different data types can still have units associated with
    them.
    """
    return hasattr(x, 'unit') and isinstance(x.unit, u.core.UnitBase)


def longdouble2string(x):
    """Convert numpy longdouble to string"""
    return repr(x)


def MJD_string2longdouble(s):
    """
    MJD_string2longdouble(s):
        Convert a MJD string to a numpy longdouble
    """
    ii, ff = s.split(".")
    return np.longfloat(ii) + np.longfloat("0."+ff)


def ddouble2ldouble(t1, t2, format='jd'):
    """
    ddouble2ldouble(t1, t2, format='jd'):
        inputs two double-precision numbers representing JD times,
        converts them to a single long double MJD value
    """
    if format == 'jd':
    # determine if the two double are JD time
        t1 = np.longdouble(t1) - np.longdouble(2400000.5)
        t = np.longdouble(t1) + np.longdouble(t2)
        return t
    else:
        t = np.longdouble([t1, t2])
    return t[0]+t[1]


def str2longdouble(str_data):
    """Return a numpy long double scalar from the input string, using strtold()
    """
    input_str = str_data.translate(maketrans('Dd', 'ee'))
    return str2ldarr1(input_str.encode())[0]


def split_prefixed_name(name):
    """A utility function that splits a prefixed name.
       Parameter
       ----------
       name : str
           Prefixed name
       Return
       ----------
       prefixPart : str
           The prefix part of the name
       indexPart : str
           The index part from the name
       indexValue : int
           The absolute index valeu
    """
    for pt in prefixPattern:
        namefield = pt.match(name)
        if namefield is None:
            continue
        prefixPart, indexPart = namefield.groups()
        if '_' in name:
            if '_' in prefixPart:
                break
            else:
                continue
        # when we have a match
        break

    if namefield is None:
        raise ValueError("Unrecognized prefix name pattern'%s'." % name)
    indexValue = int(indexPart)
    return prefixPart, indexPart, indexValue


def data2longdouble(data):
    """Return a numpy long double scalar form different type of data
       Parameters
       ---------
       data : str, ndarray, or a number
       Return
       ---------
       numpy long double type of data.
    """
    if type(data) is str:
        return str2longdouble(data)
    else:
        return np.longdouble(data)

def taylor_horner(x, coeffs):
    """Evaluate a Taylor series of coefficients at x via the Horner scheme.
    For example, if we want: 10 + 3*x/1! + 4*x^2/2! + 12*x^3/3! with
    x evaluated at 2.0, we would do:
    In [1]: taylor_horner(2.0, [10, 3, 4, 12])
    Out[1]: 40.0
    """
    result = 0.0
    if hasattr(coeffs[-1], 'unit'):
        if not hasattr(x, 'unit'):
            x = x * u.Unit("")
        result *= coeffs[-1].unit / x.unit
    fact = float(len(coeffs))
    for coeff in coeffs[::-1]:
        result = result * x / fact + coeff
        fact -= 1.0
    return result


def taylor_horner_deriv(x, coeffs):
    """Evaluate a Taylor series of coefficients at x via the Horner scheme.
    For example, if we want: 3/1! + 4*x/2! + 12*x^2/3! with
    x evaluated at 2.0, we would do:
    In [1]: taylor_horner_deriv(2.0, [10, 3, 4, 12])
    Out[1]: 15.0
    """
    result = 0.0
    if hasattr(coeffs[-1], 'unit'):
        if not hasattr(x, 'unit'):
            x = x * u.Unit("")
        result *= coeffs[-1].unit / x.unit
    fact = float(len(coeffs))
    der_coeff = float(len(coeffs)) - 1
    for coeff in coeffs[::-1]:
        result = result * x / fact + coeff * der_coeff
        fact -= 1.0
        der_coeff -= 1.0
    result = (result)/x
    return result

def is_number(s):
    """Check if it is a number string.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def create_quantization_matrix(toas, dt=1, nmin=2):
    """Create quantization matrix mapping TOAs to observing epochs."""
    isort = np.argsort(toas)

    bucket_ref = [toas[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if toas[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas[i])
            bucket_ind.append([i])

    # find only epochs with more than 1 TOA
    bucket_ind2 = [ind for ind in bucket_ind if len(ind) >= nmin]

    U = np.zeros((len(toas),len(bucket_ind2)),'d')
    for i,l in enumerate(bucket_ind2):
        U[l,i] = 1

    return U

if __name__ == "__main__":
    assert taylor_horner(2.0, [10]) == 10
    assert taylor_horner(2.0, [10, 3]) == 10 + 3*2.0
    assert taylor_horner(2.0, [10, 3, 4]) == 10 + 3*2.0 + 4*2.0**2 / 2.0
    assert taylor_horner(2.0, [10, 3, 4, 12]) == 10 + 3*2.0 + 4*2.0**2 / 2.0 + 12*2.0**3/(3.0*2.0)
