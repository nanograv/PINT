"""Miscellaneous potentially-helpful functions."""
from __future__ import absolute_import, division, print_function

import re
from contextlib import contextmanager
from six import StringIO
import numpy as np
from scipy.special import factorial
import string
import astropy.time
import astropy.units as u
import numpy as np
from astropy.time.utils import day_frac

try:
    from astropy.erfa import DJM0, d2dtf
except ImportError:
    from astropy._erfa import DJM0, d2dtf

from astropy import log
from copy import deepcopy

try:
    maketrans = str.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

if np.finfo(np.longdouble).eps > 2e-19:
    raise ValueError("This platform does not support extended precision "
                     "floating-point, and PINT cannot run without this.")

__all__ = ['PosVel', 'fortran_float', 'time_from_mjd_string',
           'time_from_longdouble', 'time_to_mjd_string',
           'time_to_mjd_string_array', 'time_to_longdouble',
           'numeric_partial', 'numeric_partials',
           'check_all_partials', 'has_astropy_unit',
           'str2longdouble',
           'PrefixError', 'split_prefixed_name', 'data2longdouble',
           'taylor_horner', 'taylor_horner_deriv', 'is_number', 'open_or_use',
           'lines_of', 'interesting_lines']

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

        if len(self.pos.shape) != len(self.vel.shape):
            # FIXME: could broadcast them, but have to be careful
            raise ValueError(
                "pos and vel must have the same number of dimensions but are {} and {}"
                .format(self.pos.shape, self.vel.shape)
            )
        elif self.pos.shape != self.vel.shape:
            self.pos, self.vel = np.broadcast_arrays(
                self.pos, self.vel, subok=True
            )

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
                raise ValueError("Attempting to add incompatible vectors: " +
                                 "%s->%s + %s->%s" % (self.origin, self.obj,
                                                      other.origin,
                                                      other.obj))

        return PosVel(self.pos + other.pos, self.vel + other.vel,
                      obj=obj, origin=origin)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __str__(self):
        if self._has_labels():
            return ("PosVel(" + str(self.pos)+", "+str(self.vel)
                    + " " + self.origin + "->" + self.obj + ")")
        else:
            return "PosVel(" + str(self.pos)+", "+str(self.vel) + ")"

    def __getitem__(self, k):
        """Allow extraction of slices of the contained arrays"""
        colon = slice(None, None, None)
        if isinstance(k, tuple):
            ix = (colon,) + k
        else:
            ix = (colon, k)
        return self.__class__(
                self.pos[ix],
                self.vel[ix],
                obj=self.obj,
                origin=self.origin)


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
    ss = s.lower().strip()
    if "e" in ss or "d" in ss:
        ss = ss.translate(maketrans("d", "e"))
        num, expon = ss.split("e")
        expon = int(expon)
        if expon < 0:
            imjd, fmjd = 0, np.longdouble(ss)
        else:
            mjd_s = num.split('.')
            # If input was given as an integer, add floating "0"
            if len(mjd_s) == 1:
                mjd_s.append("0")
            imjd_s, fmjd_s = mjd_s
            imjd = np.longdouble(int(imjd_s))
            fmjd = np.longdouble("0." + fmjd_s)
            if ss.startswith("-"):
                fmjd = -fmjd
            imjd *= 10**expon
            fmjd *= 10**expon
    else:
        mjd_s = ss.split('.')
        # If input was given as an integer, add floating "0"
        if len(mjd_s) == 1:
            mjd_s.append("0")
        imjd_s, fmjd_s = mjd_s
        imjd = int(imjd_s)
        fmjd = float("0." + fmjd_s)
        if ss.startswith("-"):
            fmjd = -fmjd
    return astropy.time.Time(imjd, fmjd, scale=scale, format='pulsar_mjd',
                             precision=9)


def time_from_longdouble(t, scale='utc'):
    st = longdouble2str(t)
    return time_from_mjd_string(st, scale)


def time_to_mjd_string(t, prec=15):
    """Print an MJD time with lots of digits (number is 'prec').

    astropy does not seem to provide this capability (yet?).
    """

    if t.format == 'pulsar_mjd':
        (imjd, fmjd) = day_frac(t.jd1 - DJM0, t.jd2)
        imjd = int(imjd)
        if fmjd < 0.0:
            imjd -= 1
        y, mo, d, hmsf = d2dtf('UTC', 9, t.jd1, t.jd2)

        if hmsf[0].size == 1:
            hmsf = np.array([list(hmsf)])
        fmjd = (hmsf[..., 0]/24.0 + hmsf[..., 1]/1440.0
                + hmsf[..., 2]/86400.0 + hmsf[..., 3]/86400.0e9)
    else:
        (imjd, fmjd) = day_frac(t.jd1 - DJM0, t.jd2)
        imjd = int(imjd)
        assert np.fabs(fmjd) < 2.0
        while fmjd >= 1.0:
            imjd += 1
            fmjd -= 1.0
        while fmjd < 0.0:
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


longdouble_mjd_eps = (70000*u.day*np.finfo(np.longdouble).eps).to(u.ns)


def time_to_longdouble(t):
    """ Return an astropy Time value as MJD in longdouble

    The returned value is accurate to within a nanosecond, while the precision of long
    double MJDs (near the present) is roughly 0.7 ns.

    """
    i_djm0 = np.longdouble(np.floor(DJM0))
    f_djm0 = np.longdouble(DJM0)-i_djm0
    return (np.longdouble(t.jd1) - i_djm0) + (np.longdouble(t.jd2)-f_djm0)


def numeric_partial(f, args, ix=0, delta=1e-6):
    """Compute the partial derivative of f numerically.

    This uses symmetric differences to estimate the partial derivative
    of a function (that takes some number of numeric arguments and may
    return an array) with respect to one of its arguments.

    """
    # r = np.array(f(*args))
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
    """Test whether x has a unit attribute containing an astropy unit.

    This is useful, because different data types can still have units
    associated with them.

    """
    return hasattr(x, 'unit') and isinstance(x.unit, u.core.UnitBase)


def longdouble2str(x):
    """Convert numpy longdouble to string."""
    return repr(x)


def str2longdouble(str_data):
    """Return a long double from the input string.

    Accepts Fortran-style exponent notation (1.0d2).

    """
    if not isinstance(str_data, str):
        raise TypeError("Need a string: {!r}".format(str_data))
    input_str = str_data.translate(maketrans('Dd', 'ee'))
    return np.longdouble(input_str.encode())


# Define prefix parameter pattern
prefix_pattern = [
    re.compile(r'^([a-zA-Z]*\d+[a-zA-Z]+)(\d+)$'),  # For the prefix like T2EFAC2
    re.compile(r'^([a-zA-Z]+)(\d+)$'),  # For the prefix like F12
    re.compile(r'^([a-zA-Z0-9]+_)(\d+)$'),  # For the prefix like DMXR1_3
    #re.compile(r'([a-zA-Z]\d[a-zA-Z]+)(\d+)'),  # for prefixes like PLANET_SHAPIRO2?
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
        if '_' in name:
            if '_' in prefix_part:
                break
            else:
                continue
        # when we have a match
        break

    if namefield is None:
        raise PrefixError("Unrecognized prefix name pattern '%s'." % name)
    return prefix_part, index_part, int(index_part)


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


def taylor_horner(x, coeffs):
    """Evaluate a Taylor series of coefficients at x via the Horner scheme.

    For example, if we want: 10 + 3*x/1! + 4*x^2/2! + 12*x^3/3! with
    x evaluated at 2.0, we would do::

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


def taylor_horner_deriv(x, coeffs, deriv_order=1):
    """Evaluate the nth derivative of a Taylor series.

    For example, if we want: first order of (10 + 3*x/1! + 4*x^2/2! + 12*x^3/3!)
    with respect to x evaluated at 2.0, we would do::

        In [1]: taylor_horner_deriv(2.0, [10, 3, 4, 12], 1)
        Out[1]: 15.0

    """
    result = 0.0
    if hasattr(coeffs[-1], 'unit'):
        if not hasattr(x, 'unit'):
            x = x * u.Unit("")
        result *= coeffs[-1].unit / x.unit
    der_coeffs = coeffs[deriv_order::]
    fact = float(len(der_coeffs))
    for coeff in der_coeffs[::-1]:
        result = result * x / fact + coeff
        fact -= 1.0
    return result


def is_number(s):
    """Check if it is a number string.

    Note that this may return False for Fortran-style floating-point
    numbers (1.0d10).

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
        cc = comments,
    for c in cc:
        cs = c.strip()
        if not cs or not c.startswith(cs):
            raise ValueError(
                "Unable to deal with comments that start with whitespace, "
                "but comment string {!r} was requested.".format(c))
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if comments is not None and ln.startswith(comments):
            continue
        yield ln

def make_toas(startMJD, endMJD, ntoas, model, freq=1400, obs='GBT'):
    '''make evenly spaced toas with residuals = 0 and  without errors
    
    might be able to do different frequencies if fed an array of frequencies,
    only works with one observatory at a time
    
    :param startMJD: starting MJD for fake toas
    :param endMJD: ending MJD for fake toas
    :param ntoas: number of fake toas to create between startMJD and endMJD
    :param model: current model
    :param freq: frequency of the fake toas, default 1400
    :param obs: observatory for fake toas, default GBT
    
    :return TOAs object with evenly spaced toas spanning given start and end MJD with ntoas toas, without errors
    '''
    #TODO:make all variables Quantity objects
    #TODO: freq default to inf
    def get_freq_array(bfv, ntoas):
        freq = np.zeros(ntoas)
        num_freqs = len(bfv)
        for ii, fv in enumerate(bfv):
            freq[ii::num_freqs] = fv
        return freq

    times = np.linspace(np.longdouble(startMJD)*u.d, np.longdouble(endMJD)*u.d, ntoas) * u.day
    freq_array = get_freq_array(np.atleast_1d(freq)*u.MHz, len(times))
    t1 = [pint.toa.TOA(t.value, obs = obs, freq=f,
                       scale=pint.observatory.get_observatory(obs).timescale) for t, f in zip(times, freq_array)]
    ts = pint.toa.TOAs(toalist=t1)
    ts.compute_TDBs()
    ts.compute_posvels()
    ts.clock_corr_info.update({'include_bipm':False,'bipm_version':'BIPM2015','include_gps':False})
    return ts

def show_param_cov_matrix(matrix,params,name='Covaraince Matrix',switchRD=False):
    '''function to print covariance matrices in a clean and easily readable way
    
    :param matrix: matrix to be printed, should be square, list of lists
    :param params: name of the parameters in the matrix, list
    :param name: title to be printed above, default Covariance Matrix
    :param switchRD: if True, switch the positions of RA and DEC to match setup of TEMPO cov. matrices
    
    :return string to be printed'''
    output = StringIO.StringIO()
    matrix = deepcopy(matrix)
    try:
        RAi = params.index('RAJ')
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
        #switch RA and DEC so cov matrix matches TEMPO
        params1[RAi:RAi+2] = [params1[RAi+1],params1[RAi]]
        i = 0
        while i < 2:
            RA = deepcopy(matrix[RAi])
            matrix[RAi] = matrix[RAi + 1]
            matrix[RAi + 1] = RA
            matrix = matrix.T
            i += 1
    output.write(name+" switch RD = "+str(switchRD)+"\n")
    output.write(' ')
    for param in params1:
        output.write('         '+param)
    i = j = 0
    while i < len(matrix):
        output.write('\n'+params1[i]+' :: ')
        while j <= i:
            num = matrix[i][j]
            if num < 0.001 and num > -0.001:
                output.write('{0: 1.2e}'.format(num)+'   : ')
            else:
                output.write('  '+'{0: 1.2f}'.format(num)+'   : ')
            j += 1
        i += 1
        j = 0
    output.write('\b:\n')
    contents = output.getvalue()
    output.close()
    return contents


if __name__ == "__main__":
    assert taylor_horner(2.0, [10]) == 10
    assert taylor_horner(2.0, [10, 3]) == 10 + 3*2.0
    assert taylor_horner(2.0, [10, 3, 4]) == 10 + 3*2.0 + 4*2.0**2 / 2.0
    assert taylor_horner(2.0, [10, 3, 4, 12]) == 10 + 3*2.0 + 4*2.0**2 / 2.0 + 12*2.0**3/(3.0*2.0)

