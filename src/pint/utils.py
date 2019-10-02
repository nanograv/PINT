"""Miscellaneous potentially-helpful functions."""
from __future__ import absolute_import, division, print_function

import decimal
import re
import string
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal

import astropy._erfa as erfa
import astropy.time
import astropy.units as u
import numpy as np
from astropy import log
from astropy._erfa import DJM0, d2dtf
from scipy.special import factorial
from six import StringIO

try:
    maketrans = str.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

# FIXME: can we make this exception raise on install instead of on use?
if np.finfo(np.longdouble).eps > 2e-19:
    raise ValueError(
        "This platform does not support extended precision "
        "floating-point, and PINT cannot run without this."
    )

__all__ = [
    "PosVel",
    "fortran_float",
    "longdouble2str",
    "str2longdouble",
    "data2longdouble",
    "time_from_longdouble",
    "time_to_longdouble",
    "time_from_mjd_string",
    "time_to_mjd_string",
    "numeric_partial",
    "numeric_partials",
    "check_all_partials",
    "has_astropy_unit",
    "PrefixError",
    "split_prefixed_name",
    "taylor_horner",
    "taylor_horner_deriv",
    "open_or_use",
    "lines_of",
    "interesting_lines",
    "mjds_to_str",
    "str_to_mjds",
    "mjds_to_jds",
    "jds_to_mjds",
    "mjds_to_jds_pulsar",
    "jds_to_mjds_pulsar",
]


# These routines are from astropy but were broken in < 3.2.2 and <= 2.0.15


def day_frac(val1, val2, factor=None, divisor=None):
    """Return the sum of ``val1`` and ``val2`` as two float64s.

    The returned floats are an integer part and the fractional remainder,
    with the latter guaranteed to be within -0.5 and 0.5 (inclusive on
    either side, as the integer is rounded to even).

    The arithmetic is all done with exact floating point operations so no
    precision is lost to rounding error.  It is assumed the sum is less
    than about 1e16, otherwise the remainder will be greater than 1.0.

    Parameters
    ----------
    val1, val2 : array of float
        Values to be summed.
    factor : float, optional
        If given, multiply the sum by it.
    divisor : float, optional
        If given, divide the sum by it.

    Returns
    -------
    day, frac : float64
        Integer and fractional part of val1 + val2.
    """
    # Add val1 and val2 exactly, returning the result as two float64s.
    # The first is the approximate sum (with some floating point error)
    # and the second is the error of the float64 sum.
    sum12, err12 = two_sum(val1, val2)

    if factor is not None:
        sum12, carry = two_product(sum12, factor)
        carry += err12 * factor
        sum12, err12 = two_sum(sum12, carry)

    if divisor is not None:
        q1 = sum12 / divisor
        p1, p2 = two_product(q1, divisor)
        d1, d2 = two_sum(sum12, -p1)
        d2 += err12
        d2 -= p2
        q2 = (d1 + d2) / divisor  # 3-part float fine here; nothing can be lost
        sum12, err12 = two_sum(q1, q2)

    # get integer fraction
    day = np.round(sum12)
    extra, frac = two_sum(sum12, -day)
    frac += extra + err12
    # Our fraction can now have gotten >0.5 or <-0.5, which means we would
    # loose one bit of precision. So, correct for that.
    excess = np.round(frac)
    day += excess
    extra, frac = two_sum(sum12, -day)
    frac += extra + err12
    return day, frac


def two_sum(a, b):
    """
    Add ``a`` and ``b`` exactly, returning the result as two float64s.
    The first is the approximate sum (with some floating point error)
    and the second is the error of the float64 sum.

    Using the procedure of Shewchuk, 1997,
    Discrete & Computational Geometry 18(3):305-363
    http://www.cs.berkeley.edu/~jrs/papers/robustr.pdf

    Returns
    -------
    sum, err : float64
        Approximate sum of a + b and the exact floating point error
    """
    x = a + b
    eb = x - a  # bvirtual in Shewchuk
    ea = x - eb  # avirtual in Shewchuk
    eb = b - eb  # broundoff in Shewchuk
    ea = a - ea  # aroundoff in Shewchuk
    return x, ea + eb


def two_product(a, b):
    """
    Multiple ``a`` and ``b`` exactly, returning the result as two float64s.
    The first is the approximate product (with some floating point error)
    and the second is the error of the float64 product.

    Uses the procedure of Shewchuk, 1997,
    Discrete & Computational Geometry 18(3):305-363
    http://www.cs.berkeley.edu/~jrs/papers/robustr.pdf

    Returns
    -------
    prod, err : float64
        Approximate product a * b and the exact floating point error
    """
    x = a * b
    ah, al = split(a)
    bh, bl = split(b)
    y1 = ah * bh
    y = x - y1
    y2 = al * bh
    y -= y2
    y3 = ah * bl
    y -= y3
    y4 = al * bl
    y = y4 - y
    return x, y


def split(a):
    """
    Split float64 in two aligned parts.

    Uses the procedure of Shewchuk, 1997,
    Discrete & Computational Geometry 18(3):305-363
    http://www.cs.berkeley.edu/~jrs/papers/robustr.pdf

    """
    c = 134217729.0 * a  # 2**27+1.
    abig = c - a
    ah = c - abig
    al = a - ah
    return ah, al


# Back to your regularly scheduled PINT


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

    Specifically, if two PosVel objects are added, the obj of one must
    equal the origin of the other (either way around). If the two
    vectors agree on both ends, then the result vector will choose the
    origin of the vector on the left.

    """

    def __init__(self, pos, vel, obj=None, origin=None):
        if len(pos) != 3:
            raise ValueError("Position vector has length %d instead of 3" % len(pos))
        if isinstance(pos, u.Quantity):
            self.pos = pos
        else:
            self.pos = np.asarray(pos)

        if len(vel) != 3:
            raise ValueError("Position vector has length %d instead of 3" % len(pos))
        if isinstance(vel, u.Quantity):
            self.vel = vel
        else:
            self.vel = np.asarray(vel)

        if len(self.pos.shape) != len(self.vel.shape):
            # FIXME: could broadcast them, but have to be careful
            raise ValueError(
                "pos and vel must have the same number of dimensions but are {} and {}".format(
                    self.pos.shape, self.vel.shape
                )
            )
        elif self.pos.shape != self.vel.shape:
            self.pos, self.vel = np.broadcast_arrays(self.pos, self.vel, subok=True)

        if bool(obj is None) != bool(origin is None):
            raise ValueError(
                "If one of obj and origin is specified, the other must be too."
            )
        self.obj = obj
        self.origin = origin
        # FIXME: what about dtype compatibility?

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
                raise ValueError(
                    "Attempting to add incompatible vectors: "
                    + "%s->%s + %s->%s"
                    % (self.origin, self.obj, other.origin, other.obj)
                )

        return PosVel(
            self.pos + other.pos, self.vel + other.vel, obj=obj, origin=origin
        )

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __str__(self):
        if self._has_labels():
            return (
                "PosVel("
                + str(self.pos)
                + ", "
                + str(self.vel)
                + " "
                + self.origin
                + "->"
                + self.obj
                + ")"
            )
        else:
            return "PosVel(" + str(self.pos) + ", " + str(self.vel) + ")"

    def __getitem__(self, k):
        """Allow extraction of slices of the contained arrays"""
        colon = slice(None, None, None)
        if isinstance(k, tuple):
            ix = (colon,) + k
        else:
            ix = (colon, k)
        return self.__class__(
            self.pos[ix], self.vel[ix], obj=self.obj, origin=self.origin
        )


def numeric_partial(f, args, ix=0, delta=1e-6):
    """Compute the partial derivative of f numerically.

    This uses symmetric differences to estimate the partial derivative
    of a function (that takes some number of numeric arguments and may
    return an array) with respect to one of its arguments.

    """
    # r = np.array(f(*args))
    args2 = list(args)
    args2[ix] = args[ix] + delta / 2.0
    r2 = np.array(f(*args2))
    args3 = list(args)
    args3[ix] = args[ix] - delta / 2.0
    r3 = np.array(f(*args3))
    return (r2 - r3) / delta


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
        # print jac
        # print njac
        d = np.abs(jac - njac) / (atol + rtol * np.abs(njac))
        print("fail fraction:", np.sum(d > 1) / float(np.sum(d >= 0)))
        worst_ix = np.unravel_index(np.argmax(d.reshape((-1,))), d.shape)
        print("max fail:", np.amax(d), "at", worst_ix)
        print("jac there:", jac[worst_ix], "njac there:", njac[worst_ix])
        raise


def has_astropy_unit(x):
    """Test whether x has a unit attribute containing an astropy unit.

    This is useful, because different data types can still have units
    associated with them.

    """
    return hasattr(x, "unit") and isinstance(x.unit, u.core.UnitBase)


# Define prefix parameter pattern
prefix_pattern = [
    re.compile(r"^([a-zA-Z]*\d+[a-zA-Z]+)(\d+)$"),  # For the prefix like T2EFAC2
    re.compile(r"^([a-zA-Z]+)(\d+)$"),  # For the prefix like F12
    re.compile(r"^([a-zA-Z0-9]+_)(\d+)$"),  # For the prefix like DMXR1_3
    # re.compile(r'([a-zA-Z]\d[a-zA-Z]+)(\d+)'),  # for prefixes like PLANET_SHAPIRO2?
]


class PrefixError(ValueError):
    pass


def split_prefixed_name(name):
    """Split a prefixed name.

    Parameters
    ----------
    name : str
       Prefixed name

    Returns
    -------
    prefixPart : str
       The prefix part of the name
    indexPart : str
       The index part from the name
    indexValue : int
       The absolute index value

    Example
    -------

        >>> split_prefixed_name("DMX_0123")
        ('DMX_', '0123', 123)
        >>> split_prefixed_name("T2EFAC17")
        ('T2EFAC', '17', 17)
        >>> split_prefixed_name("F12")
        ('F', '12', 12)
        >>> split_prefixed_name("DMXR1_2")
        ('DMXR1_', '2', 2)
        >>> split_prefixed_name("PEPOCH")
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "pint/utils.py", line 406, in split_prefixed_name
            raise PrefixError("Unrecognized prefix name pattern '%s'." % name)
        pint.utils.PrefixError: Unrecognized prefix name pattern 'PEPOCH'.

    """
    for pt in prefix_pattern:
        namefield = pt.match(name)
        if namefield is None:
            continue
        prefix_part, index_part = namefield.groups()
        if "_" in name:
            if "_" in prefix_part:
                break
            else:
                continue
        # when we have a match
        break

    if namefield is None:
        raise PrefixError("Unrecognized prefix name pattern '%s'." % name)
    return prefix_part, index_part, int(index_part)


def taylor_horner(x, coeffs):
    """Evaluate a Taylor series of coefficients at x via the Horner scheme.

    For example, if we want: 10 + 3*x/1! + 4*x^2/2! + 12*x^3/3! with
    x evaluated at 2.0, we would do::

        In [1]: taylor_horner(2.0, [10, 3, 4, 12])
        Out[1]: 40.0

    """
    result = 0.0
    if hasattr(coeffs[-1], "unit"):
        if not hasattr(x, "unit"):
            x = x * u.Unit("")
        result *= coeffs[-1].unit / x.unit
    fact = float(len(coeffs))
    for coeff in coeffs[::-1]:
        result = result * x / fact + coeff
        fact -= 1.0
    return result


def taylor_horner_deriv(x, coeffs, deriv_order=1):
    """Evaluate the nth derivative of a Taylor series.

    For example, if we want: first order of (10 + 3*x/1! + 4*x^2/2! + 12*x^3/3!)
    with respect to x evaluated at 2.0, we would do::

        In [1]: taylor_horner_deriv(2.0, [10, 3, 4, 12], 1)
        Out[1]: 15.0

    """
    result = 0.0
    if hasattr(coeffs[-1], "unit"):
        if not hasattr(x, "unit"):
            x = x * u.Unit("")
        result *= coeffs[-1].unit / x.unit
    der_coeffs = coeffs[deriv_order::]
    fact = float(len(der_coeffs))
    for coeff in der_coeffs[::-1]:
        result = result * x / fact + coeff
        fact -= 1.0
    return result


@contextmanager
def open_or_use(f, mode="r"):
    """Open a filename or use an open file.

    Specifically, if f is a string, try to use it as an argument to
    open. Otherwise just yield it. In particular anything that is not
    a subclass of ``str`` will be passed through untouched.

    """
    if isinstance(f, str):
        with open(f, mode) as fl:
            yield fl
    else:
        yield f


def lines_of(f):
    """Iterate over the lines of a file, an open file, or an iterator.

    If ``f`` is a string, try to open a file of that name. Otherwise
    treat it as an iterator and yield its values. For open files, this
    results in the lines one-by-one. For lists or other iterators it
    just yields them right through.

    """
    with open_or_use(f) as fo:
        for l in fo:
            yield l


def interesting_lines(lines, comments=None):
    """Iterate over lines skipping whitespace and comments.

    Each line has its whitespace stripped and then it is checked whether
    it .startswith(comments) . This means comments can be a string or
    a list of strings.

    """
    if comments is None:
        cc = ()
    elif isinstance(comments, tuple):
        cc = comments
    else:
        cc = (comments,)
    for c in cc:
        cs = c.strip()
        if not cs or not c.startswith(cs):
            raise ValueError(
                "Unable to deal with comments that start with whitespace, "
                "but comment string {!r} was requested.".format(c)
            )
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if comments is not None and ln.startswith(comments):
            continue
        yield ln


def show_param_cov_matrix(matrix, params, name="Covaraince Matrix", switchRD=False):
    """function to print covariance matrices in a clean and easily readable way

    :param matrix: matrix to be printed, should be square, list of lists
    :param params: name of the parameters in the matrix, list
    :param name: title to be printed above, default Covariance Matrix
    :param switchRD: if True, switch the positions of RA and DEC to match setup of TEMPO cov. matrices

    :return string to be printed"""
    output = StringIO.StringIO()
    matrix = deepcopy(matrix)
    try:
        RAi = params.index("RAJ")
    except:
        RAi = None
        switchRD = False
    params1 = []
    for param in params:
        if len(param) < 3:
            while len(param) != 3:
                param = " " + param
            params1.append(param)
        elif len(param) > 3:
            while len(param) != 3:
                param = param[:-1]
            params1.append(param)
        else:
            params1.append(param)
    if switchRD:
        # switch RA and DEC so cov matrix matches TEMPO
        params1[RAi : RAi + 2] = [params1[RAi + 1], params1[RAi]]
        i = 0
        while i < 2:
            RA = deepcopy(matrix[RAi])
            matrix[RAi] = matrix[RAi + 1]
            matrix[RAi + 1] = RA
            matrix = matrix.T
            i += 1
    output.write(name + " switch RD = " + str(switchRD) + "\n")
    output.write(" ")
    for param in params1:
        output.write("         " + param)
    i = j = 0
    while i < len(matrix):
        output.write("\n" + params1[i] + " :: ")
        while j <= i:
            num = matrix[i][j]
            if num < 0.001 and num > -0.001:
                output.write("{0: 1.2e}".format(num) + "   : ")
            else:
                output.write("  " + "{0: 1.2f}".format(num) + "   : ")
            j += 1
        i += 1
        j = 0
    output.write("\b:\n")
    contents = output.getvalue()
    output.close()
    return contents


# Precision-aware conversion functions


def fortran_float(x):
    """Convert Fortran-format floating-point strings.

    returns a copy of the input string with all 'D' or 'd' turned
    into 'e' characters.  Intended for dealing with exponential
    notation in tempo1-generated parfiles.
    """
    try:
        # First treat it as a string, wih d->e
        return float(x.translate(maketrans("Dd", "ee")))
    except AttributeError:
        # If that didn't work it may already be a numeric type
        return float(x)


def time_from_mjd_string(s, scale="utc", format="pulsar_mjd"):
    """Returns an astropy Time object generated from a MJD string input."""
    i, f = str_to_mjds(s)
    return astropy.time.Time(val=i, val2=f, scale=scale, format=format)


def time_from_longdouble(t, scale="utc", format="pulsar_mjd"):
    t = np.longdouble(t)
    i = np.floor(t)
    f = t - i
    return astropy.time.Time(val=i, val2=f, format=format, scale=scale)


def time_to_mjd_string(t):
    """Print an MJD time with lots of digits.

    astropy does not seem to provide this capability (yet?).
    """
    if t.scale == "utc" and t.format == "pulsar_mjd":
        i, f = jds_to_mjds_pulsar(t.jd1, t.jd2)
    else:
        i, f = jds_to_mjds(t.jd1, t.jd2)
    return mjds_to_str(i, f)


longdouble_mjd_eps = (70000 * u.day * np.finfo(np.longdouble).eps).to(u.ns)


def time_to_longdouble(t):
    """ Return an astropy Time value as MJD in longdouble

    The returned value is accurate to within a nanosecond, while the precision of long
    double MJDs (near the present) is roughly 0.7 ns.

    """
    if t.scale == "utc" and t.format == "pulsar_mjd":
        i, f = jds_to_mjds_pulsar(t.jd1, t.jd2)
    else:
        i, f = jds_to_mjds(t.jd1, t.jd2)
    return np.longdouble(i) + np.longdouble(f)


def data2longdouble(data):
    """Return a numpy long double scalar form different type of data

    If a string, permit Fortran-format scientific notation (1.0d2). Otherwise just use
    np.longdouble to convert. In particular if ``data`` is an array, convert all the
    elements.

    Parameters
    ----------
    data : str, np.array, or number

    Returns
    -------
    np.longdouble

    """
    if type(data) is str:
        return str2longdouble(data)
    else:
        return np.longdouble(data)


def longdouble2str(x):
    """Convert numpy longdouble to string."""
    return repr(x)


def str2longdouble(str_data):
    """Return a long double from the input string.

    Accepts Fortran-style exponent notation (1.0d2).

    """
    if not isinstance(str_data, str):
        raise TypeError("Need a string: {!r}".format(str_data))
    return np.longdouble(str_data.translate(maketrans("Dd", "ee")))


# Simplified functions: These core functions, if they can be made to work
# reliably and efficiently, should implement everything else.


def safe_kind_conversion(values, dtype):
    try:
        from collections.abc import Iterable
    except ImportError:
        from collections import Iterable
    if isinstance(values, Iterable):
        return np.asarray(values, dtype=dtype)
    else:
        return dtype(values)


# This can be removed once we only support astropy >=3.1.
# The str(c) is necessary for python2/numpy -> no unicode literals...
_new_ihmsfs_dtype = np.dtype([(str(c), np.intc) for c in "hmsf"])


def jds_to_mjds(jd1, jd2):
    return day_frac(jd1 - DJM0, jd2)


def mjds_to_jds(mjd1, mjd2):
    return day_frac(mjd1 + DJM0, mjd2)


_digits = 9


def jds_to_mjds_pulsar(jd1, jd2):
    # Do the reverse of the above calculation
    # Note this will return an incorrect value during
    # leap seconds, so raise an exception in that
    # case.
    y, mo, d, hmsf = erfa.d2dtf("UTC", _digits, jd1, jd2)
    # For ASTROPY_LT_3_1, convert to the new structured array dtype that
    # is returned by the new erfa gufuncs.
    if not hmsf.dtype.names:
        hmsf = hmsf.view(_new_ihmsfs_dtype)[..., 0]
    if np.any(hmsf["s"] == 60):
        raise ValueError(
            "UTC times during a leap second cannot be represented in pulsar_mjd format"
        )
    j1, j2 = erfa.cal2jd(y, mo, d)
    return day_frac(
        j1 - erfa.DJM0 + j2,
        hmsf["h"] / 24.0
        + hmsf["m"] / 1440.0
        + hmsf["s"] / 86400.0
        + hmsf["f"] / 86400.0 / 10 ** _digits,
    )


def mjds_to_jds_pulsar(mjd1, mjd2):
    # To get around leap second issues, first convert to YMD,
    # then back to astropy/ERFA-convention jd1,jd2 using the
    # ERFA dtf2d() routine which handles leap seconds.
    v1, v2 = day_frac(mjd1, mjd2)
    (y, mo, d, f) = erfa.jd2cal(erfa.DJM0 + v1, v2)
    # Fractional day to HMS.  Uses 86400-second day always.
    # Seems like there should be a ERFA routine for this..
    # There is: erfa.d2tf. Unfortunately it takes a "number of
    # digits" argument and returns some kind of bogus
    # fractional-part-as-an-integer thing.
    # Worse, it fails to provide nanosecond accuracy.
    # Good idea, though, because using np.remainder is
    # numerically unstable and gives bogus values now
    # and then. This is more stable.
    f *= 24
    h = safe_kind_conversion(np.floor(f), dtype=int)
    f -= h
    f *= 60
    m = safe_kind_conversion(np.floor(f), dtype=int)
    f -= m
    f *= 60
    s = f
    return erfa.dtf2d("UTC", y, mo, d, h, m, s)


# Please forgive the horrible hacks to make these work cleanly on both arrays
# and single elements


def _str_to_mjds(s):
    ss = s.lower().strip()
    if "e" in ss or "d" in ss:
        ss = ss.translate(maketrans("d", "e"))
        num, expon = ss.split("e")
        expon = int(expon)
        if expon < 0:
            imjd, fmjd = 0, np.longdouble(ss)
        else:
            mjd_s = num.split(".")
            # If input was given as an integer, add floating "0"
            if len(mjd_s) == 1:
                mjd_s.append("0")
            imjd_s, fmjd_s = mjd_s
            imjd = np.longdouble(int(imjd_s))
            fmjd = np.longdouble("0." + fmjd_s)
            if ss.startswith("-"):
                fmjd = -fmjd
            imjd *= 10 ** expon
            fmjd *= 10 ** expon
    else:
        mjd_s = ss.split(".")
        # If input was given as an integer, add floating "0"
        if len(mjd_s) == 1:
            mjd_s.append("0")
        imjd_s, fmjd_s = mjd_s
        imjd = int(imjd_s)
        fmjd = float("0." + fmjd_s)
        if ss.startswith("-"):
            fmjd = -fmjd
    return day_frac(imjd, fmjd)


def str_to_mjds(s):
    if isinstance(s, str):
        return _str_to_mjds(s)
    else:
        imjd = np.empty_like(s, dtype=int)
        fmjd = np.empty_like(s, dtype=float)
        with np.nditer(
            [s, imjd, fmjd],
            flags=["refs_ok"],
            op_flags=[["readonly"], ["writeonly"], ["writeonly"]],
        ) as it:
            for si, i, f in it:
                si = si[()]
                if not isinstance(si, str):
                    raise TypeError("Requires an array of strings")
                i[...], f[...] = _str_to_mjds(si)
            return it.operands[1], it.operands[2]


def _mjds_to_str(mjd1, mjd2):
    (imjd, fmjd) = day_frac(mjd1, mjd2)
    imjd = int(imjd)
    assert np.fabs(fmjd) < 1.0
    while fmjd < 0.0:
        imjd -= 1
        fmjd += 1.0
    assert 0 <= fmjd < 1
    return str(imjd) + "{:.16f}".format(fmjd)[1:]


_v_mjds_to_str = np.vectorize(_mjds_to_str, otypes=[np.object])


def mjds_to_str(mjd1, mjd2):
    r = _v_mjds_to_str(mjd1, mjd2)
    if r.shape == ():
        return r[()]
    else:
        return r
