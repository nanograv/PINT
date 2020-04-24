"""Miscellaneous potentially-helpful functions."""
from __future__ import absolute_import, division, print_function

import re
from contextlib import contextmanager
from copy import deepcopy
import astropy.units as u
from astropy import log
import astropy.constants as const
import numpy as np
import six
import scipy.optimize.zeros as zeros

from io import StringIO

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
    "dmxparse",
    "p_to_f",
    "pferrs",
    "weighted_mean",
    "ELL1_check",
    "mass_funct",
    "mass_funct2",
    "pulsar_mass",
    "companion_mass",
    "pulsar_age",
    "pulsar_edot",
    "pulsar_B",
    "pulsar_B_lightcyl",
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


def show_param_cov_matrix(matrix, params, name="Covariance Matrix", switchRD=False):
    """function to print covariance matrices in a clean and easily readable way

    :param matrix: matrix to be printed, should be square, list of lists
    :param params: name of the parameters in the matrix, list
    :param name: title to be printed above, default Covariance Matrix
    :param switchRD: if True, switch the positions of RA and DEC to match setup of TEMPO cov. matrices

    :return string to be printed"""
    output = StringIO()
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


def dmxparse(fitter, save=False):
    """Run dmxparse in python using PINT objects and results.

    Based off dmxparse by P. Demorest (https://github.com/nanograv/tempo/tree/master/util/dmxparse)

    Parameters
    ----------
    fitter
        PINT fitter used to get timing residuals, must have already run GLS fit
    save : bool
        if True saves output to text file in the format of the TEMPO version.
        If not output save file is desired, save = False (which is the default)
        Output file name is dmxparse.out
    
    Returns
    -------
    dictionary

        dmxs : mean-subtraced dmx values

        dmx_verrs : dmx variance errors

        dmxeps : center mjds of the dmx bins

        r1s : lower mjd bounds on the dmx bins

        r2s : upper mjd bounds on the dmx bins

        bins : dmx bins

        mean_dmx : mean dmx value

        avg_dm_err : uncertainty in average dmx

    """
    # We get the DMX values, errors, and mjds (same as in getting the DMX values for DMX v. time)
    # Get number of DMX epochs
    dmx_epochs = 0
    for p in fitter.model.params:
        if "DMX_" in p:
            dmx_epochs += 1
    # Get DMX values (will be in units of 10^-3 pc cm^-3)
    DMX_keys = []
    DMXs = []
    DMX_Errs = []
    DMX_R1 = []
    DMX_R2 = []
    DMX_center_MJD = []
    for ii in range(1, dmx_epochs + 1):
        DMX_keys.append("DMX_{:04d}".format(ii))
        DMXs.append(getattr(fitter.model, "DMX_{:04d}".format(ii)).value)
        DMX_Errs.append(
            getattr(fitter.model, "DMX_{:04d}".format(ii)).uncertainty_value
        )
        dmxr1 = getattr(fitter.model, "DMXR1_{:04d}".format(ii)).value
        dmxr2 = getattr(fitter.model, "DMXR2_{:04d}".format(ii)).value
        DMX_R1.append(dmxr1)
        DMX_R2.append(dmxr2)
        DMX_center_MJD.append((dmxr1 + dmxr2) / 2)
    DMXs = np.array(DMXs)
    DMX_Errs = np.array(DMX_Errs)
    DMX_R1 = np.array(DMX_R1)
    DMX_R2 = np.array(DMX_R2)
    DMX_center_MJD = np.array(DMX_center_MJD)

    # now get the full parameter covariance matrix from pint
    # NOTE: we will need to increase all indices by 1 to account for the 'Offset' parameter
    # that is the first index of the designmatrix
    params = np.array(list(fitter.get_fitparams().keys()))
    p_cov_mat = fitter.covariance_matrix
    # Now we get the indices that correspond to the DMX values
    DMX_p_idxs = np.zeros(dmx_epochs, dtype=int)
    for ii in range(dmx_epochs):
        DMX_p_idxs[ii] = (
            int(np.where(params == DMX_keys[ii])[0]) + 1
        )  # extra 1 is for offset parameters
    # Sort the array in numerical order for 2.7. 3.5
    DMX_p_idxs = np.sort(DMX_p_idxs)
    # Define a matrix that is just the DMX covariances
    cc = p_cov_mat[
        DMX_p_idxs[0] : DMX_p_idxs[-1] + 1, DMX_p_idxs[0] : DMX_p_idxs[-1] + 1
    ]
    n = len(DMX_Errs)
    # Find error in mean DM
    DMX_mean = np.mean(DMXs)
    DMX_mean_err = np.sqrt(cc.sum()) / float(n)
    # Do the correction for varying DM
    m = np.identity(n) - np.ones((n, n)) / float(n)
    cc = np.dot(np.dot(m, cc), m)
    DMX_vErrs = np.zeros(n)
    # We also need to correct for the units here
    for i in range(n):
        DMX_vErrs[i] = np.sqrt(cc[i, i])
    # Check we have the right number of params
    if len(DMXs) != len(DMX_Errs) or len(DMXs) != len(DMX_vErrs):
        log.error("ERROR! Number of DMX entries do not match!")
        raise RuntimeError("Number of DMX entries do not match!")

    # Output the results'
    if save:
        DMX = "DMX"
        lines = []
        lines.append("# Mean %s value = %+.6e \n" % (DMX, DMX_mean))
        lines.append("# Uncertainty in average %s = %.5e \n" % ("DM", DMX_mean_err))
        lines.append(
            "# Columns: %sEP %s_value %s_var_err %sR1 %sR2 %s_bin \n"
            % (DMX, DMX, DMX, DMX, DMX, DMX)
        )
        for k in range(dmx_epochs):
            lines.append(
                "%.4f %+.7e %.3e %.4f %.4f %s \n"
                % (
                    DMX_center_MJD[k],
                    DMXs[k] - DMX_mean,
                    DMX_vErrs[k],
                    DMX_R1[k],
                    DMX_R2[k],
                    DMX_keys[k],
                )
            )
        with open("dmxparse.out", "w") as dmxout:
            dmxout.writelines(lines)
            dmxout.close()
    # return the new mean subtracted values
    mean_sub_DMXs = DMXs - DMX_mean

    # define the output dictionary
    dmx = {}
    dmx["dmxs"] = mean_sub_DMXs
    dmx["dmx_verrs"] = DMX_vErrs
    dmx["dmxeps"] = DMX_center_MJD
    dmx["r1s"] = DMX_R1
    dmx["r2s"] = DMX_R2
    dmx["bins"] = DMX_keys
    dmx["mean_dmx"] = DMX_mean
    dmx["avg_dm_err"] = DMX_mean_err

    return dmx


def weighted_mean(arrin, weights_in, inputmean=None, calcerr=False, sdev=False):
    """Compute weighted mean of input values

    Calculate the weighted mean, error, and optionally standard deviation of
    an input array.  By default error is calculated assuming the weights are
    1/err^2, but if you send calcerr=True this assumption is dropped and the
    error is determined from the weighted scatter.

    Parameters
    ----------
    arrin : array
    Array containing the numbers whose weighted mean is desired.      
    weights: array
    A set of weights for each element in array. For measurements with 
    uncertainties, these should be 1/sigma^2.
    inputmean: float, optional
        An input mean value, around which the mean is calculated.
    calcerr : bool, optional
        Calculate the weighted error.  By default the error is calculated as
        1/sqrt( weights.sum() ).  If calcerr=True it is calculated as 
        sqrt((w**2 * (arr-mean)**2).sum() )/weights.sum().
    sdev : bool, optional
        If True, also return the weighted standard deviation as a third
        element in the tuple. Defaults to False.

    Returns
    -------
    wmean, werr: tuple
    A tuple of the weighted mean and error. If sdev=True the
    tuple will also contain sdev: wmean,werr,wsdev

    Notes
    -----
    Converted from IDL: 2006-10-23. Erin Sheldon, NYU
    Copied from PRESTO to PINT : 2020-04-18

   """
    arr = arrin
    weights = weights_in
    wtot = weights.sum()
    # user has input a mean value
    if inputmean is None:
        wmean = (weights * arr).sum() / wtot
    else:
        wmean = float(inputmean)
    # how should error be calculated?
    if calcerr:
        werr2 = (weights ** 2 * (arr - wmean) ** 2).sum()
        werr = np.sqrt(werr2) / wtot
    else:
        werr = 1.0 / np.sqrt(wtot)
    # should output include the weighted standard deviation?
    if sdev:
        wvar = (weights * (arr - wmean) ** 2).sum() / wtot
        wsdev = np.sqrt(wvar)
        return wmean, werr, wsdev
    else:
        return wmean, werr


def p_to_f(p, pd, pdd=None):
    """Converts P, Pdot to F, Fdot (or vice versa)

    Convert period, period derivative and period second
    derivative to the equivalent frequency counterparts.
    Will also convert from f to p.
    """
    f = 1.0 / p
    fd = -pd / (p * p)
    if pdd is None:
        return [f, fd]
    else:
        if pdd == 0.0:
            fdd = 0.0
        else:
            fdd = 2.0 * pd * pd / (p ** 3.0) - pdd / (p * p)
        return [f, fd, fdd]


def pferrs(porf, porferr, pdorfd=None, pdorfderr=None):
    """Convert P, Pdot to F, Fdot with uncertainties.

    Calculate the period or frequency errors and
    the pdot or fdot errors from the opposite one.
    """
    if pdorfd is None:
        return [1.0 / porf, porferr / porf ** 2.0]
    else:
        forperr = porferr / porf ** 2.0
        fdorpderr = np.sqrt(
            (4.0 * pdorfd ** 2.0 * porferr ** 2.0) / porf ** 6.0
            + pdorfderr ** 2.0 / porf ** 4.0
        )
        [forp, fdorpd] = p_to_f(porf, pdorfd)
        return [forp, forperr, fdorpd, fdorpderr]


def pulsar_age(f, fdot, n=3, fo=1e99 * u.Hz):
    """Compute pulsar characteristic age

    Return the age of a pulsar given the spin frequency
    and frequency derivative.  By default, the characteristic age
    is returned (assuming a braking index 'n'=3 and an initial
    spin frequency fo >> f).  But 'n' and 'fo' can be set.
    """
    return (-f / ((n - 1.0) * fdot) * (1.0 - (f / fo) ** (n - 1.0))).to(u.yr)


def pulsar_edot(f, fdot, I=1.0e45 * u.g * u.cm ** 2):
    """Compute pulsar spindown energy loss rate

    Return the pulsar Edot (in erg/s) given the spin frequency and
    frequency derivative. The NS moment of inertia is assumed to be
    I = 1.0e45 g cm^2 by default.
    """
    return (-4.0 * np.pi ** 2 * I * f * fdot).to(u.erg / u.s)


def pulsar_B(f, fdot):
    """Compute pulsar surface magnetic field
    
    Return the estimated pulsar surface magnetic field strength
    given the spin frequency and frequency derivative.
    """
    # This is a hack to use the traditional formula by stripping the units.
    # It would be nice to improve this to a  proper formula with units
    return 3.2e19 * u.G * np.sqrt(-fdot.to(u.Hz / u.s).value / f.to(u.Hz).value ** 3.0)


def pulsar_B_lightcyl(f, fdot):
    """Compute pulsar magnetic field at the light cylinder
    
    Return the estimated pulsar magnetic field strength at the
    light cylinder given the spin frequency and
    frequency derivative.
    """
    p, pd = p_to_f(f, fdot)
    # This is a hack to use the traditional formula by stripping the units.
    # It would be nice to improve this to a  proper formula with units
    return (
        2.9e8
        * u.G
        * p.to(u.s).value ** (-5.0 / 2.0)
        * np.sqrt(pd.to(u.dimensionless_unscaled).value)
    )


def mass_funct(pb, x):
    """Compute binary mass function from period and semi-major axis

    Parameters
    ----------
    pb : Quantity
        Binary period
    x : Quantity in `pint.ls`
        Semi-major axis, A1SINI, in units of ls

    Returns
    -------
    f_m : Quantity
        Mass function in solar masses
    """
    fm = 4.0 * np.pi ** 2 * x ** 3 / (const.G * pb ** 2)
    return fm.to(u.solMass)


def mass_funct2(mp, mc, i):
    """Compute binary mass function from masses and inclination

    Parameters
    ----------
    mp : Quantity
        Pulsar mass, typically in u.solMass
    mc : Quantity
        Companion mass, typically in u.solMass
    i : Angle
        Inclination angle, in u.deg or u.rad

    Notes
    -----
    Inclination is such that edge on is `i = 90*u.deg`
    An 'average' orbit has cos(i) = 0.5, or `i = 60*u.deg`
    """
    return (mc * np.sin(i)) ** 3.0 / (mc + mp) ** 2.0


def pulsar_mass(pb, x, mc, inc):
    """Compute pulsar mass from orbit and Shapiro delay parameters

    Return the pulsar mass (in solar mass units) for a binary.
    Finds the value using a bisection technique.
 
    Parameters
    ----------
    pb : Quantity
        Binary orbital period
    x : Quantity
        Projected semi-major axis (aka ASINI) in `pint.ls`
    mc : Quantity
        Companion mass in u.solMass
    inc : Angle
        Inclination angle, in u.deg or u.rad
    """
    massfunct = mass_funct(pb, x)

    # Do some unit manipulation here so that scipy bisect doesn't see the units
    def localmf(mp, mc=mc, mf=massfunct, i=inc):
        return (mass_funct2(mp * u.solMass, mc, i) - mf).value

    return zeros.bisect(localmf, 0.0, 1000.0)


def companion_mass(pb, x, inc=60.0 * u.deg, mpsr=1.4 * u.solMass):
    """Commpute the companion mass from the orbital parameters

    Compute companion mass for a binary system from orbital mechanics,
    not Shapiro delay.

    Parameters
    ----------
    pb : Quantity
        Binary orbital period
    x : Quantity
        Projected semi-major axis (aka ASINI) in `pint.ls`
    inc : Angle, optional
        Inclination angle, in u.deg or u.rad. Default is 60 deg.
    mpsr : Quantity, optional
        Pulsar mass in u.solMass. Default is 1.4 Msun
    """
    massfunct = mass_funct(pb, x)

    # Do some unit manipulation here so that scipy bisect doesn't see the units
    def localmf(mc, mp=mpsr, mf=massfunct, i=inc):
        return (mass_funct2(mp, mc * u.solMass, i) - mf).value

    return zeros.bisect(localmf, 0.001, 1000.1)


def ELL1_check(A1, E, TRES, NTOA, outstring=True):
    """Check for validity of assumptions in ELL1 binary model

    Checks whether the assumptions that allow ELL1 to be safely used are 
    satisfied. To work properly, we should have:
    asini/c * ecc**2 << timing precision / sqrt(# TOAs)
    or A1 * E**2 << TRES / sqrt(NTOA)

    Parameters
    ----------
    A1 : Quantity
        Projected semi-major axis (aka ASINI) in `pint.ls`
    E : Quantity (dimensionless)
        Eccentricity
    TRES : Quantity
        RMS TOA uncertainty
    NTOA : int
        Number of TOAs in the fit
    outstring : bool, optional

    Returns
    -------
    bool or str
        Returns True if ELL1 is safe to use, otherwise False.
        If outstring is True then returns a string summary instead.

    """
    lhs = A1 / const.c * E ** 2.0
    rhs = TRES / np.sqrt(NTOA)
    if outstring:
        s = "Checking applicability of ELL1 model -- \n"
        s += "    Condition is asini/c * ecc**2 << timing precision / sqrt(# TOAs) to use ELL1\n"
        s += "    asini/c * ecc**2    = {:.3g} \n".format(lhs.to(u.us))
        s += "    TRES / sqrt(# TOAs) = {:.3g} \n".format(rhs.to(u.us))
    if lhs * 50.0 < rhs:
        if outstring:
            s += "    Should be fine.\n"
            return s
        return True
    elif lhs * 5.0 < rhs:
        if outstring:
            s += "    Should be OK, but not optimal.\n"
            return s
        return True
    else:
        if outstring:
            s += "    *** WARNING*** Should probably use BT or DD instead!\n"
            return s
        return False
