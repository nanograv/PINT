"""Miscellaneous potentially-helpful functions."""
from __future__ import absolute_import, division, print_function

import re
from contextlib import contextmanager
from copy import deepcopy

import astropy.units as u
import numpy as np
import six
from six import StringIO

__all__ = [
    "PosVel",
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
    "show_param_cov_matrix",
]


# Actual exported tools


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
        try:
            prefix_part, index_part = pt.match(name).groups()
            break
        except AttributeError:
            continue
    else:
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
    if isinstance(f, six.string_types):
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
    elif isinstance(comments, six.string_types):
        cc = (comments,)
    else:
        cc = tuple(comments)
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
        if ln.startswith(cc):
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
