"""Miscellaneous potentially-helpful functions.

Warning
-------
Functions:

- :func:`~pint.derived_quantities.a1sini`
- :func:`~pint.derived_quantities.companion_mass`
- :func:`~pint.derived_quantities.gamma`
- :func:`~pint.derived_quantities.mass_funct`
- :func:`~pint.derived_quantities.mass_funct2`
- :func:`~pint.derived_quantities.omdot`
- :func:`~pint.derived_quantities.omdot_to_mtot`
- :func:`~pint.derived_quantities.p_to_f`
- :func:`~pint.derived_quantities.pbdot`
- :func:`~pint.derived_quantities.pferrs`
- :func:`~pint.derived_quantities.pulsar_B`
- :func:`~pint.derived_quantities.pulsar_B_lightcyl`
- :func:`~pint.derived_quantities.pulsar_age`
- :func:`~pint.derived_quantities.pulsar_edot`
- :func:`~pint.derived_quantities.pulsar_mass`
- :func:`~pint.derived_quantities.shklovskii_factor`

have moved to :mod:`pint.derived_quantities`.

- :func:`pint.simulation.calculate_random_models`

has moved to :mod:`pint.simulation`.

"""
import configparser
import datetime
import getpass
import os
import platform
import re
import textwrap
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO
from warnings import warn
from loguru import logger as log

import astropy.constants as const
import astropy.coordinates as coords
import astropy.coordinates.angles as angles
import astropy.units as u
import numpy as np
import scipy.optimize.zeros as zeros
from scipy.special import fdtrc

import pint
import pint.pulsar_ecliptic

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
    "pmtot",
    "dmxparse",
    "dmxstats",
    "dmx_ranges_old",
    "dmx_ranges",
    "weighted_mean",
    "ELL1_check",
    "FTest",
    "add_dummy_distance",
    "remove_dummy_distance",
    "info_string",
    "print_color_examples",
    "colorize",
]

COLOR_NAMES = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
TEXT_ATTRIBUTES = [
    "normal",
    "bold",
    "subdued",
    "italic",
    "underscore",
    "blink",
    "reverse",
    "concealed",
]

# Actual exported tools


class PosVel:
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

    Parameters
    ----------
    x: astropy.units.Quantity
        Input value; may be an array.
    coeffs: list of astropy.units.Quantity
        Coefficient array; must have length at least one. The coefficient in
        position ``i`` is multiplied by ``x**i``. Each coefficient should
        just be a number, not an array. The units should be compatible once
        multiplied by an appropriate power of x.

    Returns
    -------
    astropy.units.Quantity
        Output value; same shape as input. Units as inferred from inputs.
    """
    return taylor_horner_deriv(x, coeffs, deriv_order=0)


def taylor_horner_deriv(x, coeffs, deriv_order=1):
    """Evaluate the nth derivative of a Taylor series.

    For example, if we want: first order of (10 + 3*x/1! + 4*x^2/2! + 12*x^3/3!)
    with respect to x evaluated at 2.0, we would do::

        In [1]: taylor_horner_deriv(2.0, [10, 3, 4, 12], 1)
        Out[1]: 15.0

    Parameters
    ----------
    x: astropy.units.Quantity
        Input value; may be an array.
    coeffs: list of astropy.units.Quantity
        Coefficient array; must have length at least one. The coefficient in
        position ``i`` is multiplied by ``x**i``. Each coefficient should
        just be a number, not an array. The units should be compatible once
        multiplied by an appropriate power of x.
    deriv_order: int
        The order of the derivative to take (that is, how many times to differentiate).
        Must be non-negative.

    Returns
    -------
    astropy.units.Quantity
        Output value; same shape as input. Units as inferred from inputs.
    """
    result = 0.0
    if hasattr(coeffs[-1], "unit"):
        if not hasattr(x, "unit"):
            x = x * u.Unit("")
        result *= coeffs[-1].unit / x.unit
    der_coeffs = coeffs[deriv_order::]
    fact = len(der_coeffs)
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
    if isinstance(f, (str, bytes)):
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
    elif isinstance(comments, (str, bytes)):
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


def pmtot(model):
    """Compute and return the total proper motion from a model object

    Calculates total proper motion from the parameters of the model, in either
    equatorial or ecliptic coordinates.  Note that in both cases, pulsar timing
    codes define the proper motion in the longitude coordinate to be the
    the actual angular rate of change of position on the sky rather than the change in coordinate value,
    so PMRA = (d(RAJ)/dt)*cos(DECJ). This is different from the astrometry community where mu_alpha = d(alpha)/dt.
    Thus, we don't need to include cos(DECJ) or cos(ELAT) in our calculation.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel

    Returns
    -------
    pmtot : astropy.units.Quantity
        Returns total proper motion with units of ``u.mas/u.yr``

    Raises
    ------
    AttributeError
        If no Astrometry component is found in the model
    """

    if "AstrometryEcliptic" in model.components.keys():
        return np.sqrt(model.PMELONG.quantity**2 + model.PMELAT.quantity**2).to(
            u.mas / u.yr
        )
    elif "AstrometryEquatorial" in model.components.keys():
        return np.sqrt(model.PMRA.quantity**2 + model.PMDEC.quantity**2).to(
            u.mas / u.yr
        )
    else:
        raise AttributeError("No Astrometry component found")


class dmxrange:
    """Internal class for building DMX ranges"""

    def __init__(self, lofreqs, hifreqs):
        """lofreqs and hifreqs are lists of MJDs that are in the low or high band respectively"""
        self.los = lofreqs
        self.his = hifreqs
        self.min = min(lofreqs + hifreqs) - 0.001 * u.d
        self.max = max(lofreqs + hifreqs) + 0.001 * u.d

    def sum_print(self):
        print(
            "{:8.2f}-{:8.2f} ({:8.2f}): NLO={:5d} NHI={:5d}".format(
                self.min.value,
                self.max.value,
                self.max - self.min,
                len(self.los),
                len(self.his),
            )
        )


def dmx_ranges_old(
    toas,
    divide_freq=1000.0 * u.MHz,
    offset=0.01 * u.d,
    max_diff=15.0 * u.d,
    verbose=False,
):
    """Compute initial DMX ranges for a set of TOAs

    This is a rudimentary translation of $TEMPO/utils/dmx_ranges/DMX_ranges2.py

    Parameters
    ----------
    divide_freq : Quantity, MHz
        Requires TOAs above and below this freq for a good DMX range
    offset : Quantity, days
        The buffer to include around each DMX range. Warning, may cause bins to overlap?!?
    max_diff : Quantity, days
        Maximum duration of a DMX bin
    verbose : bool
        If True, print out verbose information about the DMX ranges including par file lines.

    Returns
    -------
    mask : bool array
        Array with True for all TOAs that got assigned to a DMX bin
    component : TimingModel.Component object
        A DMX Component class with the DMX ranges included
    """
    import pint.models.parameter
    from pint.models.timing_model import Component

    MJDs = toas.get_mjds()
    freqs = toas.table["freq"]

    loMJDs = MJDs[freqs < divide_freq]
    hiMJDs = MJDs[freqs > divide_freq]
    # Round off the dates to 0.1 days and only keep unique values so we ignore closely spaced TOAs
    loMJDs = np.unique(loMJDs.round(1))
    hiMJDs = np.unique(hiMJDs.round(1))
    log.info("There are {} dates with freqs > {} MHz".format(len(hiMJDs), divide_freq))
    log.info(
        "There are {} dates with freqs < {} MHz\n".format(len(loMJDs), divide_freq)
    )

    DMXs = []

    good_his = set([])
    bad_los = []
    # Walk through all of the low freq obs
    for ii, loMJD in enumerate(loMJDs):
        # find all the high freq obs within max_diff days
        # of the low freq obs
        hi_close = hiMJDs[np.fabs(hiMJDs - loMJD) < max_diff]
        # and where they are closer to this loMJD compared to the
        # other nearby ones
        if ii > 0:
            diffs = np.fabs(hi_close - loMJD)
            lodiffs = np.fabs(hi_close - loMJDs[ii - 1])
            hi_close = hi_close[diffs < lodiffs]
        if ii < len(loMJDs) - 1:
            diffs = np.fabs(hi_close - loMJD)
            hidiffs = np.fabs(hi_close - loMJDs[ii + 1])
            hi_close = hi_close[diffs < hidiffs]
        if len(hi_close):  # add a DMXrange
            DMXs.append(dmxrange([loMJD], list(hi_close)))
            good_his = good_his.union(set(hi_close))
        else:
            bad_los.append(loMJD)

    bad_los = set(bad_los)
    saved_los = []
    # print bad_los
    # Now walk through the DMXs and see if we can't fit a bad_lo freq in
    for bad_lo in bad_los:
        absmindiff = 2 * max_diff
        ind = 0
        for ii, DMX in enumerate(DMXs):
            if (
                np.fabs(bad_lo - DMX.min) < max_diff
                and np.fabs(bad_lo - DMX.max) < max_diff
            ):
                mindiff = min(np.fabs(bad_lo - DMX.min), np.fabs(bad_lo - DMX.max))
                if mindiff < absmindiff:
                    absmindiff = mindiff
                    ind = ii
        if absmindiff < max_diff:
            # print DMXs[ind].min, DMXs[ind].max, bad_lo
            DMXs[ind].los.append(bad_lo)
            # update the min and max vals
            DMXs[ind].min = min(DMXs[ind].los + DMXs[ind].his)
            DMXs[ind].max = max(DMXs[ind].los + DMXs[ind].his)
            saved_los.append(bad_lo)

    # These are the low-freq obs we can't save
    bad_los -= set(saved_los)
    bad_los = sorted(list(bad_los))

    # These are the high-freq obs we can't save
    bad_his = set(hiMJDs) - good_his
    bad_his = sorted(list(bad_his))

    if verbose:
        print("\n These are the 'good' ranges for DMX and days are low/high freq:")
        for DMX in DMXs:
            DMX.sum_print()

        print("\nRemove high-frequency data from these days:")
        for hibad in bad_his:
            print("{:8.2f}".format(hibad.value))
        print("\nRemove low-frequency data from these days:")
        for lobad in bad_los:
            print("{:8.2f}".format(lobad.value))

        print("\n Enter the following in your parfile")
        print("-------------------------------------")
        print("DMX         {:.2f}".format(max_diff.value))
        oldmax = 0.0
        for ii, DMX in enumerate(DMXs):
            print("DMX_{:04d}      0.0       {}".format(ii + 1, 1))
            print("DMXR1_{:04d}      {:10.4f}".format(ii + 1, (DMX.min - offset).value))
            print("DMXR2_{:04d}      {:10.4f}".format(ii + 1, (DMX.max + offset).value))
            if DMX.min < oldmax:
                print("Ack!  This shouldn't be happening!")
            oldmax = DMX.max
    # Init mask to all False
    mask = np.zeros_like(MJDs.value, dtype=bool)
    # Mark TOAs as True if they are in any DMX bin
    for DMX in DMXs:
        mask[np.logical_and(MJDs > DMX.min - offset, MJDs < DMX.max + offset)] = True
    log.info("{} out of {} TOAs are in a DMX bin".format(mask.sum(), len(mask)))
    # Instantiate a DMX component
    dmx_class = Component.component_types["DispersionDMX"]
    dmx_comp = dmx_class()
    # Add parameters
    for ii, DMX in enumerate(DMXs):
        if ii == 0:
            # Already have DMX_0001 in component, so just set parameters
            dmx_comp.DMX_0001.value = 0.0
            dmx_comp.DMX_0001.frozen = False
            dmx_comp.DMXR1_0001.value = (DMX.min - offset).value
            dmx_comp.DMXR2_0001.value = (DMX.max + offset).value

        else:
            # Add the DMX parameters
            dmx_par = pint.models.parameter.prefixParameter(
                parameter_type="float",
                name="DMX_{:04d}".format(ii + 1),
                value=0.0,
                units=u.pc / u.cm**3,
                frozen=False,
            )
            dmx_comp.add_param(dmx_par, setup=True)

            dmxr1_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR1_{:04d}".format(ii + 1),
                value=(DMX.min - offset).value,
                units=u.d,
            )
            dmx_comp.add_param(dmxr1_par, setup=True)

            dmxr2_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR2_{:04d}".format(ii + 1),
                value=(DMX.max + offset).value,
                units=u.d,
            )
            dmx_comp.add_param(dmxr2_par, setup=True)
    # Validate component
    dmx_comp.validate()

    return mask, dmx_comp


def dmx_ranges(toas, divide_freq=1000.0 * u.MHz, binwidth=15.0 * u.d, verbose=False):
    """Compute initial DMX ranges for a set of TOAs

    This is an alternative algorithm for computing DMX ranges

    Parameters
    ----------
    divide_freq : Quantity, MHz
        Requires TOAs above and below this freq for a good DMX range
    offset : Quantity, days
        The buffer to include around each DMX range. Warning, may cause bins to overlap?!?
    max_diff : Quantity, days
        Maximum duration of a DMX bin
    verbose : bool
        If True, print out verbose information about the DMX ranges including par file lines.

    Returns
    -------
    mask : bool array
        Array with True for all TOAs that got assigned to a DMX bin
    component : TimingModel.Component object
        A DMX Component class with the DMX ranges included
    """
    import pint.models.parameter
    from pint.models.timing_model import Component

    MJDs = toas.get_mjds()
    freqs = toas.table["freq"].quantity

    DMXs = []

    prevbinR2 = MJDs[0] - 0.001 * u.d
    while True:
        # Consider all TOAs with times after the last bin up through a total span of binwidth
        # Get indexes that should be in this bin
        # If there are no more MJDs to process, we are done.
        if not np.any(MJDs > prevbinR2):
            break
        startMJD = MJDs[MJDs > prevbinR2][0]
        binidx = np.logical_and(MJDs > prevbinR2, MJDs <= startMJD + binwidth)
        if not np.any(binidx):
            break
        binMJDs = MJDs[binidx]
        binfreqs = freqs[binidx]
        loMJDs = binMJDs[binfreqs < divide_freq]
        hiMJDs = binMJDs[binfreqs >= divide_freq]
        # If we have freqs below and above the divide, this is a good bin
        if np.any(binfreqs < divide_freq) and np.any(binfreqs > divide_freq):
            DMXs.append(dmxrange(list(loMJDs), list(hiMJDs)))
        else:
            # These TOAs cannot be used
            pass
        prevbinR2 = binMJDs.max()

    if verbose:
        print(
            "\n These are the good DMX ranges with number of TOAs above/below the dividing freq:"
        )
        for DMX in DMXs:
            DMX.sum_print()

    # Init mask to all False
    mask = np.zeros_like(MJDs.value, dtype=bool)
    # Mark TOAs as True if they are in any DMX bin
    for DMX in DMXs:
        mask[np.logical_and(MJDs >= DMX.min, MJDs <= DMX.max)] = True
    log.info("{} out of {} TOAs are in a DMX bin".format(mask.sum(), len(mask)))
    # Instantiate a DMX component
    dmx_class = Component.component_types["DispersionDMX"]
    dmx_comp = dmx_class()
    # Add parameters
    for ii, DMX in enumerate(DMXs):
        if ii == 0:
            # Already have DMX_0001 in component, so just set parameters
            dmx_comp.DMX_0001.value = 0.0
            dmx_comp.DMX_0001.frozen = False
            dmx_comp.DMXR1_0001.value = DMX.min.value
            dmx_comp.DMXR2_0001.value = DMX.max.value

        else:
            # Add the DMX parameters
            dmx_par = pint.models.parameter.prefixParameter(
                parameter_type="float",
                name="DMX_{:04d}".format(ii + 1),
                value=0.0,
                units=u.pc / u.cm**3,
                frozen=False,
            )
            dmx_comp.add_param(dmx_par, setup=True)

            dmxr1_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR1_{:04d}".format(ii + 1),
                value=DMX.min.value,
                units=u.d,
            )
            dmx_comp.add_param(dmxr1_par, setup=True)

            dmxr2_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR2_{:04d}".format(ii + 1),
                value=DMX.max.value,
                units=u.d,
            )
            dmx_comp.add_param(dmxr2_par, setup=True)
    # Validate component
    dmx_comp.validate()

    return mask, dmx_comp


def dmxstats(fitter):
    """Run dmxparse in python using PINT objects and results.

    Based off dmxparse by P. Demorest (https://github.com/nanograv/tempo/tree/master/util/dmxparse)

    Parameters
    ----------
    fitter
        PINT fitter used to get timing residuals, must have already run GLS fit
    """

    model = fitter.model
    mjds = fitter.toas.get_mjds()
    freqs = fitter.toas.table["freq"]
    ii = 1
    while hasattr(model, "DMX_{:04d}".format(ii)):
        mmask = np.logical_and(
            mjds.value > getattr(model, "DMXR1_{:04d}".format(ii)).value,
            mjds.value < getattr(model, "DMXR2_{:04d}".format(ii)).value,
        )
        if np.any(mmask):
            mjds_in_bin = mjds[mmask]
            freqs_in_bin = freqs[mmask]
            span = (mjds_in_bin.max() - mjds_in_bin.min()).to(u.d)
            # Warning: min() and max() seem to strip the units
            freqspan = freqs_in_bin.max() - freqs_in_bin.min()
            print(
                "DMX_{:04d}: NTOAS={:5d}, MJDSpan={:14.4f}, FreqSpan={:8.3f}-{:8.3f}".format(
                    ii, mmask.sum(), span, freqs_in_bin.min(), freqs_in_bin.max()
                )
            )
        else:
            print(
                "DMX_{:04d}: NTOAS={:5d}, MJDSpan={:14.4f}, FreqSpan={:8.3f}-{:8.3f}".format(
                    ii, mmask.sum(), 0 * u.d, 0 * u.MHz, 0 * u.MHz
                )
            )
        ii += 1

    return


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

    Raises
    ------
    RuntimeError : If the model has no DMX parameters, or if there is a parsing problem

    """
    # We get the DMX values, errors, and mjds (same as in getting the DMX values for DMX v. time)
    # Get number of DMX epochs
    dmx_epochs = []
    for p in fitter.model.params:
        if "DMX_" in p:
            dmx_epochs.append(p.split("_")[-1])
    # Check to make sure that there are DMX values in the model
    if not dmx_epochs:
        raise RuntimeError("No DMX values in model!")
    # Get DMX values (will be in units of 10^-3 pc cm^-3)
    DMX_keys = []
    DMXs = []
    DMX_Errs = []
    DMX_R1 = []
    DMX_R2 = []
    DMX_center_MJD = []
    mask_idxs = []
    for ii in dmx_epochs:
        DMX_keys.append("DMX_{:}".format(ii))
        DMXs.append(getattr(fitter.model, "DMX_{:}".format(ii)).value)
        mask_idxs.append(getattr(fitter.model, "DMX_{:}".format(ii)).frozen)
        DMX_Errs.append(getattr(fitter.model, "DMX_{:}".format(ii)).uncertainty_value)
        dmxr1 = getattr(fitter.model, "DMXR1_{:}".format(ii)).value
        dmxr2 = getattr(fitter.model, "DMXR2_{:}".format(ii)).value
        DMX_R1.append(dmxr1)
        DMX_R2.append(dmxr2)
        DMX_center_MJD.append((dmxr1 + dmxr2) / 2)
    DMXs = np.array(DMXs)
    DMX_Errs = np.array(DMX_Errs)
    DMX_R1 = np.array(DMX_R1)
    DMX_R2 = np.array(DMX_R2)
    DMX_center_MJD = np.array(DMX_center_MJD)
    # If any value need to be masked, do it
    if True in mask_idxs:
        log.warning(
            "Some DMX bins were not fit for, masking these bins for computation."
        )
        DMX_Errs = np.ma.array(DMX_Errs, mask=mask_idxs)
        DMX_keys_ma = np.ma.array(DMX_keys, mask=mask_idxs)
    else:
        DMX_keys_ma = None

    # Make sure that the fitter has a covariance matrix, otherwise return the initial values
    if hasattr(fitter, "parameter_covariance_matrix"):
        # now get the full parameter covariance matrix from pint
        # access by label name to make sure we get the right values
        # make sure they are sorted in ascending order
        cc = fitter.parameter_covariance_matrix.get_label_matrix(
            sorted(["DMX_" + x for x in dmx_epochs])
        )
        n = len(DMX_Errs) - np.sum(mask_idxs)
        # Find error in mean DM
        DMX_mean = np.mean(DMXs)
        DMX_mean_err = np.sqrt(cc.matrix.sum()) / float(n)
        # Do the correction for varying DM
        m = np.identity(n) - np.ones((n, n)) / float(n)
        cc = np.dot(np.dot(m, cc.matrix), m)
        DMX_vErrs = np.zeros(n)
        # We also need to correct for the units here
        for i in range(n):
            DMX_vErrs[i] = np.sqrt(cc[i, i])
        # If array was masked, we need to add values back in where they were masked
        if DMX_keys_ma is not None:
            # Only need to add value to DMX_vErrs
            DMX_vErrs = np.insert(DMX_vErrs, np.where(mask_idxs)[0], None)
    else:
        log.warning(
            "Fitter does not have covariance matrix, returning values from model"
        )
        DMX_mean = np.mean(DMXs)
        DMX_mean_err = np.mean(DMX_Errs)
        DMX_vErrs = DMX_Errs
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
        for k in range(len(dmx_epochs)):
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

    # Get units to multiply returned arrays by
    DMX_units = getattr(fitter.model, "DMX_{:}".format(dmx_epochs[0])).units
    DMXR_units = getattr(fitter.model, "DMXR1_{:}".format(dmx_epochs[0])).units

    # define the output dictionary
    dmx = {}
    dmx["dmxs"] = mean_sub_DMXs * DMX_units
    dmx["dmx_verrs"] = DMX_vErrs * DMX_units
    dmx["dmxeps"] = DMX_center_MJD * DMXR_units
    dmx["r1s"] = DMX_R1 * DMXR_units
    dmx["r2s"] = DMX_R2 * DMXR_units
    dmx["bins"] = DMX_keys
    dmx["mean_dmx"] = DMX_mean * DMX_units
    dmx["avg_dm_err"] = DMX_mean_err * DMX_units

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
        werr2 = (weights**2 * (arr - wmean) ** 2).sum()
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


@u.quantity_input
def ELL1_check(
    A1: u.cm, E: u.dimensionless_unscaled, TRES: u.us, NTOA: int, outstring=True
):
    """Check for validity of assumptions in ELL1 binary model

    Checks whether the assumptions that allow ELL1 to be safely used are
    satisfied. To work properly, we should have:
    :math:`asini/c  e^2 \ll {\\rm timing precision} / \sqrt N_{\\rm TOA}`
    or :math:`A1 E^2 \ll TRES / \sqrt N_{\\rm TOA}`

    Parameters
    ----------
    A1 : astropy.units.Quantity
        Projected semi-major axis (aka ASINI) in `pint.ls`
    E : astropy.units.Quantity (dimensionless)
        Eccentricity
    TRES : astropy.units.Quantity
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
    lhs = A1 / const.c * E**2.0
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


def FTest(chi2_1, dof_1, chi2_2, dof_2):
    """Run F-test.

    Compute an F-test to see if a model with extra parameters is
    significant compared to a simpler model.  The input values are the
    (non-reduced) chi^2 values and the numbers of DOF for '1' the
    original model and '2' for the new model (with more fit params).
    The probability is computed exactly like Sherpa's F-test routine
    (in Ciao) and is also described in the Wikipedia article on the
    F-test:  http://en.wikipedia.org/wiki/F-test
    The returned value is the probability that the improvement in
    chi2 is due to chance (i.e. a low probability means that the
    new fit is quantitatively better, while a value near 1 means
    that the new model should likely be rejected).

    Parameters
    -----------
    chi2_1 : float
        Chi-squared value of model with fewer parameters
    dof_1 : int
        Degrees of freedom of model with fewer parameters
    chi2_2 : float
        Chi-squared value of model with more parameters
    dof_2 : int
        Degrees of freedom of model with more parameters

    Returns
    --------
    ft : float
        F-test significance value for the model with the larger number of
        components over the other.
    """
    delta_chi2 = chi2_1 - chi2_2
    if delta_chi2 > 0 and dof_1 != dof_2:
        delta_dof = dof_1 - dof_2
        new_redchi2 = chi2_2 / dof_2
        F = float((delta_chi2 / delta_dof) / new_redchi2)  # fdtr doesn't like float128
        ft = fdtrc(delta_dof, dof_2, F)
    elif dof_1 == dof_2:
        log.warning("Models have equal degrees of freedom, cannot perform F-test.")
        ft = np.nan
    elif delta_chi2 <= 0:
        log.warning(
            "Chi^2 for Model 2 is larger than Chi^2 for Model 1, cannot perform F-test."
        )
        ft = 1.0
    else:
        raise ValueError(
            f"Mystery problem in Ftest - maybe NaN? {chi2_1} {dof_1} {chi2_2} {dof_2}"
        )
    return ft


def add_dummy_distance(c, distance=1 * u.kpc):
    """Adds a dummy distance to a SkyCoord object for applying proper motion

    Parameters
    ----------
    c: astropy.coordinates.SkyCoord
        current SkyCoord object without distance but with proper motion and obstime
    distance: astropy.units.Quantity, optional
        distance to supply

    Returns
    -------
    cnew : astropy.coordinates.SkyCoord
        new SkyCoord object with a distance attached
    """

    if c.frame.data.differentials == {}:
        log.warning(
            "No proper motions available for %r: returning coordinates unchanged" % c
        )
        return c
    if c.obstime is None:
        log.warning("No obstime available for %r: returning coordinates unchanged" % c)
        return c

    if isinstance(c.frame, coords.builtin_frames.icrs.ICRS):
        if hasattr(c, "pm_ra_cosdec"):
            cnew = coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra_cosdec,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                distance=distance,
                frame=coords.ICRS,
            )
        else:
            # it seems that after applying proper motions
            # it changes the RA pm to pm_ra instead of pm_ra_cosdec
            # although the value seems the same
            cnew = coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                distance=distance,
                frame=coords.ICRS,
            )

        return cnew
    elif isinstance(c.frame, coords.builtin_frames.galactic.Galactic):
        cnew = coords.SkyCoord(
            l=c.l,
            b=c.b,
            pm_l_cosb=c.pm_l_cosb,
            pm_b=c.pm_b,
            obstime=c.obstime,
            distance=distance,
            frame=coords.Galactic,
        )
        return cnew
    elif isinstance(c.frame, pint.pulsar_ecliptic.PulsarEcliptic):
        cnew = coords.SkyCoord(
            lon=c.lon,
            lat=c.lat,
            pm_lon_coslat=c.pm_lon_coslat,
            pm_lat=c.pm_lat,
            obstime=c.obstime,
            distance=distance,
            obliquity=c.obliquity,
            frame=pint.pulsar_ecliptic.PulsarEcliptic,
        )
        return cnew
    else:
        log.warning(
            "Do not know coordinate frame for %r: returning coordinates unchanged" % c
        )
        return c


def remove_dummy_distance(c):
    """Removes a dummy distance from a SkyCoord object after applying proper motion

    Parameters
    ----------
    c: astropy.coordinates.SkyCoord
        current SkyCoord object with distance and with proper motion and obstime

    Returns
    -------
    cnew : astropy.coordinates.SkyCoord
        new SkyCoord object with a distance removed
    """

    if c.frame.data.differentials == {}:
        log.warning(
            "No proper motions available for %r: returning coordinates unchanged" % c
        )
        return c
    if c.obstime is None:
        log.warning("No obstime available for %r: returning coordinates unchanged" % c)
        return c

    if isinstance(c.frame, coords.builtin_frames.icrs.ICRS):
        if hasattr(c, "pm_ra_cosdec"):

            cnew = coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra_cosdec,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                frame=coords.ICRS,
            )
        else:
            # it seems that after applying proper motions
            # it changes the RA pm to pm_ra instead of pm_ra_cosdec
            # although the value seems the same
            cnew = coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                frame=coords.ICRS,
            )
        return cnew
    elif isinstance(c.frame, coords.builtin_frames.galactic.Galactic):
        cnew = coords.SkyCoord(
            l=c.l,
            b=c.b,
            pm_l_cosb=c.pm_l_cosb,
            pm_b=c.pm_b,
            obstime=c.obstime,
            frame=coords.Galactic,
        )
        return cnew
    elif isinstance(c.frame, pint.pulsar_ecliptic.PulsarEcliptic):
        cnew = coords.SkyCoord(
            lon=c.lon,
            lat=c.lat,
            pm_lon_coslat=c.pm_lon_coslat,
            pm_lat=c.pm_lat,
            obstime=c.obstime,
            obliquity=c.obliquity,
            frame=pint.pulsar_ecliptic.PulsarEcliptic,
        )
        return cnew
    else:
        log.warning(
            "Do not know coordinate frame for %r: returning coordinates unchanged" % c
        )
        return c


def info_string(prefix_string="# ", comment=None):
    """Returns an informative string about the current state of PINT.

    Adds:

    * Creation date
    * PINT version
    * Username (given by the `gitpython`_ global configuration ``user.name``
      if available, in addition to :func:`getpass.getuser`).
    * Host (given by :func:`platform.node`)
    * OS (given by :func:`platform.platform`)
    * plus a user-supplied comment (if present).

    Parameters
    ----------
    prefix_string: str, default='# '
        a string to be prefixed to the output (often to designate as a
        comment or similar)
    comment: str, optional
        a free-form comment string to be included if present

    Returns
    -------
    str
        informative string

    Examples
    --------
    >>> import pint.utils
    >>> print(pint.utils.info_string(prefix_string="# ",comment="Example comment"))
    # Created: 2021-07-21T09:39:45.606894
    # PINT_version: 0.8.2+311.ge351099d
    # User: David Kaplan (dlk)
    # Host: margle-2.local
    # OS: macOS-10.14.6-x86_64-i386-64bit
    # Comment: Example comment

    Multi-line comments are allowed:

    >>> import pint.utils
    >>> print(pint.utils.info_string(prefix_string="C ",
    ...                              comment="Example multi-line comment\\nAlso using a different comment character"))
    C Created: 2021-07-21T09:40:34.172333
    C PINT_version: 0.8.2+311.ge351099d
    C User: David Kaplan (dlk)
    C Host: margle-2.local
    C OS: macOS-10.14.6-x86_64-i386-64bit
    C Comment: Example multi-line comment
    C Comment: Also using a different comment character

    Full example of writing a par and tim file:

    >>> from pint.models import get_model_and_toas
    >>> # the locations of these may vary
    >>> timfile = "tests/datafile/NGC6440E.tim"
    >>> parfile = "tests/datafile/NGC6440E.par"
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> print(m.as_parfile(comment="Here is a comment on the par file"))
    # Created: 2021-07-22T08:24:27.101479
    # PINT_version: 0.8.2+439.ge81c9b11.dirty
    # User: David Kaplan (dlk)
    # Host: margle-2.local
    # OS: macOS-10.14.6-x86_64-i386-64bit
    # Comment: Here is a comment on the par file
    PSR                            1748-2021E
    EPHEM                               DE421
    CLK                             UTC(NIST)
    ...

    >>> from pint.models import get_model_and_toas
    >>> import io
    >>> # the locations of these may vary
    >>> timfile = "tests/datafile/NGC6440E.tim"
    >>> parfile = "tests/datafile/NGC6440E.par"
    >>> m, t = get_model_and_toas(parfile, timfile)
    >>> f = io.StringIO(parfile)
    >>> t.write_TOA_file(f, comment="Here is a comment on the tim file")
    >>> f.seek(0)
    >>> print(f.getvalue())
    FORMAT 1
    C Created: 2021-07-22T08:24:27.213529
    C PINT_version: 0.8.2+439.ge81c9b11.dirty
    C User: David Kaplan (dlk)
    C Host: margle-2.local
    C OS: macOS-10.14.6-x86_64-i386-64bit
    C Comment: Here is a comment on the tim file
    unk 1949.609000 53478.2858714192189005 21.710 gbt  -format Princeton -ddm 0.0
    unk 1949.609000 53483.2767051885165973 21.950 gbt  -format Princeton -ddm 0.0
    unk 1949.609000 53489.4683897879295023 29.950 gbt  -format Princeton -ddm 0.0
    ....


    Notes
    -----
    This can be called via  :func:`~pint.toa.TOAs.write_TOA_file` on a :class:`~~pint.toa.TOAs` object,
    or :func:`~pint.models.timing_model.TimingModel.as_parfile` on a
    :class:`~pint.models.timing_model.TimingModel` object.

    .. _gitpython: https://gitpython.readthedocs.io/en/stable/
    """
    # try to get the git user if defined
    try:
        import git

        # user-level git config
        c = git.GitConfigParser()
        username = c.get_value("user", option="name") + f" ({getpass.getuser()})"
    except (configparser.NoOptionError, configparser.NoSectionError, ImportError):
        username = getpass.getuser()

    s = f"""
    Created: {datetime.datetime.now().isoformat()}
    PINT_version: {pint.__version__}
    User: {username}
    Host: {platform.node()}
    OS: {platform.platform()}
    """

    s = textwrap.dedent(s)
    # remove blank lines
    s = os.linesep.join([x for x in s.splitlines() if x])
    if comment is not None:
        if os.linesep in comment:
            s += os.linesep + os.linesep.join(
                [f"Comment: {x}" for x in comment.splitlines()]
            )
        else:
            s += f"{os.linesep}Comment: {comment}"

    if (prefix_string is not None) and (len(prefix_string) > 0):
        s = os.linesep.join([prefix_string + x for x in s.splitlines()])
    return s


def list_parameters(class_=None):
    """List parameters understood by PINT.

    Parameters
    ----------
    class_: type, optional
        If provided, produce a list of parameters understood by the Component type; if None,
        return a list of parameters understood by all Components known to PINT.

    Returns
    -------
    list of dict
        Each entry is a dictionary describing one parameter. Dictionary values are all strings
        or lists of strings, and will include at least "name", "classes", and "description".
    """
    if class_ is not None:
        from pint.models.parameter import (
            boolParameter,
            intParameter,
            maskParameter,
            prefixParameter,
            strParameter,
        )

        result = []
        inst = class_()
        for p in inst.params:
            pm = getattr(inst, p)
            d = dict(
                name=pm.name,
                class_=f"{class_.__module__}.{class_.__name__}",
                description=pm.description,
            )
            if pm.aliases:
                d["aliases"] = [a for a in pm.aliases if a != pm.name]
            if pm.units:
                d["kind"] = pm.units.to_string()
                if not d["kind"]:
                    d["kind"] = "number"
            elif isinstance(pm, boolParameter):
                d["kind"] = "boolean"
            elif isinstance(pm, strParameter):
                d["kind"] = "string"
            elif isinstance(pm, intParameter):
                d["kind"] = "integer"
            if isinstance(pm, prefixParameter):
                d["name"] = pm.prefix + "{number}"
                d["aliases"] = [a + "{number}" for a in pm.prefix_aliases]
            if isinstance(pm, maskParameter):
                d["name"] = pm.origin_name + " {flag} {value}"
                d["aliases"] = [a + " {flag} {value}" for a in pm.prefix_aliases]
            if "aliases" in d and not d["aliases"]:
                del d["aliases"]
            result.append(d)
        return result
    else:
        import pint.models.timing_model

        results = {}
        ct = pint.models.timing_model.Component.component_types.copy()
        ct["TimingModel"] = pint.models.timing_model.TimingModel
        for v in ct.values():
            for d in list_parameters(v):
                n = d["name"]
                class_ = d.pop("class_")
                if n not in results:
                    d["classes"] = [class_]
                    results[n] = d
                else:
                    r = results[n].copy()
                    r.pop("classes")
                    if r != d:
                        raise ValueError(
                            f"Parameter {d} in class {class_} does not match {results[n]}"
                        )
                    results[n]["classes"].append(class_)
        return sorted(results.values(), key=lambda d: d["name"])


def colorize(text, fg_color, bg_color=None, attribute=None):
    """Colorizes a string (including unicode strings) for printing on the terminal

    For an example of usage, as well as a demonstration as to what the
    attributes and colors look like, check out :func:`~pint.utils.print_color_examples`

    Parameters
    ----------
    text : string
        The text to colorize. Can include unicode.
    fg_color : _type_
        Foreground color name. The color names (fg or bg) are one of:
        'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan',
        or 'white'.
    bg_color : _type_, optional
        Background color name, by default None. Same choices as for `fg_color`.
    attribute : _type_, optional
        Text attribute, by default None. The text attributes are one of:
        'normal', 'bold', 'subdued', 'italic', 'underscore', 'blink',
        'reverse', or 'concealed'.

    Returns
    -------
    string
        The colorized string using the defined text attribute.
    """
    COLOR_FORMAT = "\033[%dm\033[%d;%dm%s\033[0m"
    FOREGROUND = dict(zip(COLOR_NAMES, list(range(30, 38))))
    BACKGROUND = dict(zip(COLOR_NAMES, list(range(40, 48))))
    ATTRIBUTE = dict(zip(TEXT_ATTRIBUTES, [0, 1, 2, 3, 4, 5, 7, 8]))
    fg = FOREGROUND.get(fg_color, 39)
    bg = BACKGROUND.get(bg_color, 49)
    att = ATTRIBUTE.get(attribute, 0)
    return COLOR_FORMAT % (att, bg, fg, text)


def print_color_examples():
    """Print example terminal colors and attributes for/using :func:`~pint.utils.colorize`"""
    for att in TEXT_ATTRIBUTES:
        for fg in COLOR_NAMES:
            for bg in COLOR_NAMES:
                print(
                    colorize(f"{fg:>8} {att:<11}", fg, bg_color=bg, attribute=att),
                    end="",
                )
            print("")
