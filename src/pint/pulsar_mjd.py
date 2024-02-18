"""PulsarMJD special time format.

MJDs seem simple but if you want to use them with the UTC time scale you have
to deal with the fact that every so often a UTC day is either 86401 or 86399
seconds long. :class:`astropy.time.Time` has a policy on how to interpret MJDs
in UTC, and in that time scale all times can be expressed in MJD, but the rate
at which MJDs advance is not one day per 86400 SI seconds. (This technique,
when applied to UNIX time, is called a "leap_smear_" and is used by all Google
APIs.) This is not how (some? all?) observatories construct the MJDs they
record; observatories record times by converting UNIX time to MJDs in a way
that ignores leap seconds; this means that there is more than one time that
will produce the same MJD on a leap second day. (Observatories usually use a
maser to keep very accurate time, but this is used only to identify the
beginnings of seconds; NTP is used to determine which second to record.) We
therefore introduce the "pulsar_mjd" time format to capture the way in which
our data actually occurs.

An MJD expressed in the "pulsar_mjd" time scale will never occur during a
leap second. No negative leap seconds have yet been inserted, that is,
all days have been either 86400 or 86401 seconds long. If a negative leap
second does occur, it is not totally clear what will happen if an MJD
is provided that corresponds to a nonexistent time.

This is not a theoretical consideration: at least one pulsar observer
was observing the sky at the moment a leap second was introduced.

.. _leap_smear: https://developers.google.com/time/smear
"""
import warnings

import erfa
import astropy.time
import astropy.units as u
import numpy as np
from astropy.time import Time
from astropy.time.formats import TimeFormat

try:
    maketrans = str.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans


# This check is implemented in pint.utils, but we want to avoid circular imports
if np.finfo(np.longdouble).eps > 2e-19:
    import warnings

    def readable_warning(message, category, filename, lineno, line=None):
        return "%s: %s\n" % (category.__name__, message)

    warnings.formatwarning = readable_warning

    msg = (
        "This platform does not support extended precision "
        "floating-point, and PINT will run at reduced precision."
    )
    warnings.warn(msg, RuntimeWarning)


__all__ = [
    "Time",
    "PulsarMJD",
    "MJDLong",
    "PulsarMJDLong",
    "MJDString",
    "PulsarMJDString",
    "fortran_float",
    "longdouble2str",
    "str2longdouble",
    "data2longdouble",
    "time_from_longdouble",
    "time_to_longdouble",
    "time_from_mjd_string",
    "time_to_mjd_string",
    "mjds_to_str",
    "str_to_mjds",
    "mjds_to_jds",
    "jds_to_mjds",
    "mjds_to_jds_pulsar",
    "jds_to_mjds_pulsar",
]


class PulsarMJD(TimeFormat):
    """Change handling of days with leap seconds.

    MJD using tempo/tempo2 convention for time within leap second days.
    This is only relevant if scale='utc', otherwise will act like the
    standard astropy MJD time format.
    """

    name = "pulsar_mjd"

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)
        # for times before the first leap second we don't need to do anything
        # it's annoying to handle parts-of-arrays like this though
        if self._scale == "utc":
            self.jd1, self.jd2 = mjds_to_jds_pulsar(val1, val2)
        else:
            self.jd1, self.jd2 = mjds_to_jds(val1, val2)

    @property
    def value(self):
        if self._scale == "utc":
            mjd1, mjd2 = jds_to_mjds_pulsar(self.jd1, self.jd2)
        else:
            mjd1, mjd2 = jds_to_mjds(self.jd1, self.jd2)

        return mjd1 + mjd2


class MJDLong(TimeFormat):
    """Support conversion of MJDs to and from extended precision."""

    name = "mjd_long"

    def _check_val_type(self, val1, val2):
        if val1.dtype != np.longdouble:
            raise ValueError(
                "mjd_long requires a long double number but got {!r} of type {}".format(
                    val1, val1.dtype
                )
            )
        if val2 is None:
            val2 = 0
        elif val2.dtype != np.longdouble:
            raise ValueError(
                "mjd_long requires a long double number but got {!r} of type {}".format(
                    val2, val2.dtype
                )
            )
        return val1, val2

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)

        i = np.floor(val1)
        f = val1 - i
        i2 = np.floor(val2)
        f2 = val2 - i2

        self.jd1, self.jd2 = mjds_to_jds((i + i2).astype(float), (f + f2).astype(float))

    @property
    def value(self):
        mjd1, mjd2 = jds_to_mjds(self.jd1, self.jd2)
        return np.longdouble(mjd1) + np.longdouble(mjd2)


class PulsarMJDLong(TimeFormat):
    """Support conversion of pulsar MJDs to and from extended precision."""

    name = "pulsar_mjd_long"

    def _check_val_type(self, val1, val2):
        if val1.dtype != np.longdouble:
            raise ValueError(
                "pulsar_mjd_long requires a long double number but got {!r} of type {}".format(
                    val1, val1.dtype
                )
            )
        if val2 is None:
            val2 = 0
        elif val2.dtype != np.longdouble:
            raise ValueError(
                "pulsar_mjd_long requires a long double number but got {!r} of type {}".format(
                    val2, val2.dtype
                )
            )
        return val1, val2

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)

        i = np.floor(val1)
        f = val1 - i
        i2 = np.floor(val2)
        f2 = val2 - i2

        i = (i + i2).astype(float)
        f = (f + f2).astype(float)

        if self._scale == "utc":
            self.jd1, self.jd2 = mjds_to_jds_pulsar(i, f)
        else:
            self.jd1, self.jd2 = mjds_to_jds(i, f)

    @property
    def value(self):
        if self._scale == "utc":
            mjd1, mjd2 = jds_to_mjds_pulsar(self.jd1, self.jd2)
            return mjd1 + np.longdouble(mjd2)
        else:
            mjd1, mjd2 = jds_to_mjds(self.jd1, self.jd2)
            return np.longdouble(mjd1) + np.longdouble(mjd2)


class MJDString(TimeFormat):
    """Support full-accuracy reading and writing of MJDs in string form."""

    name = "mjd_string"

    def _check_val_type(self, val1, val2):
        if val1.dtype.kind not in "US":
            raise ValueError("mjd_string requires a string but got {!r}".format(val1))
        if val2 is not None:
            raise ValueError(
                "mjd_string doesn't accept a val2 but got {!r}".format(val2)
            )
        return val1, val2

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)
        self.jd1, self.jd2 = mjds_to_jds(*str_to_mjds(val1))

    @property
    def value(self):
        return mjds_to_str(*jds_to_mjds(self.jd1, self.jd2))


class PulsarMJDString(TimeFormat):
    """Support full-accuracy reading and writing of pulsar MJDs in string form."""

    name = "pulsar_mjd_string"

    def _check_val_type(self, val1, val2):
        if val1.dtype.kind not in "US":
            raise ValueError(
                "pulsar_mjd_string requires a string but got {!r}".format(val1)
            )
        if val2 is not None:
            raise ValueError(
                "pulsar_mjd_string doesn't accept a val2 but got {!r}".format(val2)
            )
        return val1, val2

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)
        if self._scale == "utc":
            self.jd1, self.jd2 = mjds_to_jds_pulsar(*str_to_mjds(val1))
        else:
            self.jd1, self.jd2 = mjds_to_jds(*str_to_mjds(val1))

    @property
    def value(self):
        if self._scale == "utc":
            return mjds_to_str(*jds_to_mjds_pulsar(self.jd1, self.jd2))
        else:
            return mjds_to_str(*jds_to_mjds(self.jd1, self.jd2))


def time_from_mjd_string(s, scale="utc", format="pulsar_mjd"):
    """Returns an astropy Time object generated from a MJD string input."""
    if format.lower().startswith("pulsar_mjd"):
        return astropy.time.Time(val=s, scale=scale, format="pulsar_mjd_string")
    elif format.lower().startswith("mjd"):
        return astropy.time.Time(val=s, scale=scale, format="mjd_string")
    else:
        raise ValueError(f"Format {format} is not recognizable as an MJD format")


def time_from_longdouble(t, scale="utc", format="pulsar_mjd"):
    t = np.longdouble(t)
    i = float(np.floor(t))
    f = float(t - i)
    return astropy.time.Time(val=i, val2=f, format=format, scale=scale)


def time_to_mjd_string(t):
    """Print an MJD time with lots of digits.

    astropy does not seem to provide this capability (yet?).
    """
    if t.format.startswith("pulsar_mjd"):
        return t.pulsar_mjd_string
    else:
        return t.mjd_string


longdouble_mjd_eps = (70000 * u.day * np.finfo(np.longdouble).eps).to(u.ns)


def time_to_longdouble(t):
    """Return an astropy Time value as MJD in longdouble

    The returned value is accurate to within a nanosecond, while the precision of long
    double MJDs (near the present) is roughly 0.7 ns.

    """
    return t.pulsar_mjd_long if t.format.startswith("pulsar_mjd") else t.mjd_long


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
    return str2longdouble(data) if type(data) is str else np.longdouble(data)


def quantity2longdouble_withunit(data):
    """Return a astropy.units.Quantity containing a numpy long double scalar from a different dtype

    Parameters
    ----------
    data : astropy.units.Quantity

    Returns
    -------
    astropy.units.Quantity

    See also
    --------
    :func:data2longdouble

    """
    unit = data.unit
    data = np.longdouble(data.to_value(unit))
    return data * unit


def longdouble2str(x):
    """Convert numpy longdouble to string."""
    return repr(x)


def str2longdouble(str_data):
    """Return a long double from the input string.

    Accepts Fortran-style exponent notation (1.0d2).
    """
    if not isinstance(str_data, (str, bytes)):
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
    return day_frac(jd1 - erfa.DJM0, jd2)


def mjds_to_jds(mjd1, mjd2):
    return day_frac(mjd1 + erfa.DJM0, mjd2)


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
    if np.any((hmsf["s"] == 60) & (hmsf["f"] != 0)):
        # if f is exactly zero, this is probably fine to treat as the end of the day.
        raise ValueError(
            "UTC times during a leap second cannot be represented in pulsar_mjd format"
        )
    j1, j2 = erfa.cal2jd(y, mo, d)
    return day_frac(
        j1 - erfa.DJM0 + j2,
        hmsf["h"] / 24.0
        + hmsf["m"] / 1440.0
        + hmsf["s"] / 86400.0
        + hmsf["f"] / 86400.0 / 10**_digits,
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
            fmjd = np.longdouble(f"0.{fmjd_s}")
            if ss.startswith("-"):
                fmjd = -fmjd
            imjd *= 10**expon
            fmjd *= 10**expon
    else:
        mjd_s = ss.split(".")
        # If input was given as an integer, add floating "0"
        if len(mjd_s) == 1:
            mjd_s.append("0")
        imjd_s, fmjd_s = mjd_s
        imjd = int(imjd_s)
        fmjd = float(f"0.{fmjd_s}")
        if ss.startswith("-"):
            fmjd = -fmjd
    return day_frac(imjd, fmjd)


def str_to_mjds(s):
    if isinstance(s, (str, bytes)):
        return _str_to_mjds(s)
    imjd = np.empty_like(s, dtype=int)
    fmjd = np.empty_like(s, dtype=float)
    with np.nditer(
        [s, imjd, fmjd],
        flags=["refs_ok"],
        op_flags=[["readonly"], ["writeonly"], ["writeonly"]],
    ) as it:
        for si, i, f in it:
            si = si[()]
            if not isinstance(si, (str, bytes)):
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
    # return str(imjd) + str(1+fmjd)[1:]


_v_mjds_to_str = np.vectorize(_mjds_to_str, otypes=[np.dtype("U30")])


def mjds_to_str(mjd1, mjd2):
    r = _v_mjds_to_str(mjd1, mjd2)
    return r[()] if r.shape == () else r


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
