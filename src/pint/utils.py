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
import hashlib
import os
import platform
import re
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from scipy.optimize import minimize
from numdifftools import Hessian

import astropy.constants as const
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy import constants
from astropy.time import Time
from loguru import logger as log
from scipy.special import fdtrc
from scipy.linalg import cho_factor, cho_solve
from copy import deepcopy
import warnings

import pint
import pint.pulsar_ecliptic
from pint.toa_select import TOASelect


__all__ = [
    "PINTPrecisionError",
    "check_longdouble_precision",
    "require_longdouble_precision",
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
    "dmxrange",
    "sum_print",
    "dmx_ranges_old",
    "dmx_ranges",
    "dmxselections",
    "xxxselections",
    "dmxstats",
    "dmxparse",
    "get_prefix_timerange",
    "get_prefix_timeranges",
    "find_prefix_bytime",
    "merge_dmx",
    "split_dmx",
    "split_swx",
    "wavex_setup",
    "translate_wave_to_wavex",
    "get_wavex_freqs",
    "get_wavex_amps",
    "translate_wavex_to_wave",
    "weighted_mean",
    "ELL1_check",
    "FTest",
    "add_dummy_distance",
    "remove_dummy_distance",
    "info_string",
    "list_parameters",
    "colorize",
    "print_color_examples",
    "group_iterator",
    "compute_hash",
    "get_conjunction",
    "divide_times",
    "convert_dispersion_measure",
    "parse_time",
    "get_unit",
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


class PINTPrecisionError(RuntimeError):
    pass


# A warning is emitted in pint.pulsar_mjd if sufficient precision is not available


def check_longdouble_precision():
    """Check whether long doubles have adequate precision.

    Returns True if long doubles have enough precision to use PINT
    for sub-microsecond timing on this machine.
    """
    return np.finfo(np.longdouble).eps < 2e-19


def require_longdouble_precision():
    """Raise an exception if long doubles do not have enough precision.

    Raises RuntimeError if PINT cannot be run with high precision on this
    machine.
    """
    if not check_longdouble_precision():
        raise PINTPrecisionError(
            "PINT needs higher precision floating point than you have available. PINT uses the numpy longdouble type to represent modified Julian days, and this machine does not have sufficient numerical precision to represent sub-microsecond times with np.longdouble. On an M1 Mac you will need to use a Rosetta environment, or on a Windows machine you will need to us a different Python interpreter. Some PINT operations can work with reduced precision, but you have requested one that cannot."
        )


class PosVel:
    """Position/Velocity class.

    The class is used to represent the 6 values describing position
    and velocity vectors.  Instances have 'pos' and 'vel' attributes
    that are numpy arrays of floats (and can have attached astropy
    units).  The 'pos' and 'vel' params are 3-vectors of the positions
    and velocities respectively.

    The coordinates are generally assumed to be aligned with ICRF (J2000),
    i.e. they are in an inertial, not earth-rotating frame

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
            raise ValueError(f"Position vector has length {len(pos)} instead of 3")
        self.pos = pos if isinstance(pos, u.Quantity) else np.asarray(pos)

        if len(vel) != 3:
            raise ValueError(f"Position vector has length {len(pos)} instead of 3")
        self.vel = vel if isinstance(vel, u.Quantity) else np.asarray(vel)

        if len(self.pos.shape) != len(self.vel.shape):
            # FIXME: could broadcast them, but have to be careful
            raise ValueError(
                f"pos and vel must have the same number of dimensions but are {self.pos.shape} and {self.vel.shape}"
            )

        elif self.pos.shape != self.vel.shape:
            self.pos, self.vel = np.broadcast_arrays(self.pos, self.vel, subok=True)

        if (obj is None) != (origin is None):
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
                    f"Attempting to add incompatible vectors: {self.origin}->{self.obj} + {other.origin}->{other.obj}"
                )

        return PosVel(
            self.pos + other.pos, self.vel + other.vel, obj=obj, origin=origin
        )

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __str__(self):
        return (
            f"PosVel({str(self.pos)}, {str(self.vel)} {self.origin}->{self.obj})"
            if self._has_labels()
            else f"PosVel({str(self.pos)}, {str(self.vel)})"
        )

    def __getitem__(self, k):
        """Allow extraction of slices of the contained arrays"""
        colon = slice(None, None, None)
        ix = (colon,) + k if isinstance(k, tuple) else (colon, k)
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
    re.compile(r"^([a-zA-Z]+)0*(\d+)$"),  # For the prefix like F12
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
        raise PrefixError(f"Unrecognized prefix name pattern '{name}'.")
    return prefix_part, index_part, int(index_part)


def taylor_horner(x, coeffs):
    """Evaluate a Taylor series of coefficients at x via the Horner scheme.

    For example, if we want: 10 + 3*x/1! + 4*x^2/2! + 12*x^3/3! with
    x evaluated at 2.0, we would do::

        In [1]: taylor_horner(2.0, [10, 3, 4, 12])
        Out[1]: 40.0

    Parameters
    ----------
    x: float or numpy.ndarray or astropy.units.Quantity
        Input value; may be an array.
    coeffs: list of astropy.units.Quantity or uncertainties.ufloat
        Coefficient array; must have length at least one. The coefficient in
        position ``i`` is multiplied by ``x**i``. Each coefficient should
        just be a number, not an array. The units should be compatible once
        multiplied by an appropriate power of x.

    Returns
    -------
    float or numpy.ndarray or astropy.units.Quantity
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
    x: float or numpy.ndarray or astropy.units.Quantity
        Input value; may be an array.
    coeffs: list of astropy.units.Quantity or uncertainties.ufloat
        Coefficient array; must have length at least one. The coefficient in
        position ``i`` is multiplied by ``x**i``. Each coefficient should
        just be a number, not an array. The units should be compatible once
        multiplied by an appropriate power of x.
    deriv_order: int
        The order of the derivative to take (that is, how many times to differentiate).
        Must be non-negative.

    Returns
    -------
    float or numpy.ndarray or astropy.units.Quantity
        Output value; same shape as input. Units as inferred from inputs.
    """
    assert deriv_order >= 0
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
    if isinstance(f, (str, bytes, Path)):
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
        yield from fo


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
    log.info(f"There are {len(hiMJDs)} dates with freqs > {divide_freq} MHz")
    log.info(f"There are {len(loMJDs)} dates with freqs < {divide_freq} MHz\n")

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
    log.info(f"{mask.sum()} out of {len(mask)} TOAs are in a DMX bin")
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
    while np.any(MJDs > prevbinR2):
        # Consider all TOAs with times after the last bin up through a total span of binwidth
        # Get indexes that should be in this bin
        # If there are no more MJDs to process, we are done.
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
    log.info(f"{mask.sum()} out of {len(mask)} TOAs are in a DMX bin")
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


def xxxselections(model, toas, prefix="DM"):
    """Map DMX/SWX/other selections to TOAs

    Parameters
    ----------
    model : pint.models.TimingModel
    toas : pint.toa.TOAs
    prefix : str
        Name of selection

    Returns
    -------
    dict :
        keys are XXX indices, values are the TOAs selected for each index
    """
    if not any(p.startswith(f"{prefix}X") for p in model.params):
        return {}
    toas_selector = TOASelect(is_range=True)
    X_mapping = model.get_prefix_mapping(f"{prefix}X_")
    XR1_mapping = model.get_prefix_mapping(f"{prefix}XR1_")
    XR2_mapping = model.get_prefix_mapping(f"{prefix}XR2_")
    condition = {}
    for ii in X_mapping:
        r1 = getattr(model, XR1_mapping[ii]).quantity
        r2 = getattr(model, XR2_mapping[ii]).quantity
        condition[X_mapping[ii]] = (r1.mjd, r2.mjd)
    return toas_selector.get_select_index(condition, toas["mjd_float"])


def dmxselections(model, toas):
    """Map DMX selections to TOAs

    Parameters
    ----------
    model : pint.models.TimingModel
    toas : pint.toa.TOAs

    Returns
    -------
    dict :
        keys are DMX indices, values are the TOAs selected for each index
    """
    toas_selector = TOASelect(is_range=True)
    DMX_mapping = model.get_prefix_mapping("DMX_")
    DMXR1_mapping = model.get_prefix_mapping("DMXR1_")
    DMXR2_mapping = model.get_prefix_mapping("DMXR2_")
    condition = {}
    for ii in DMX_mapping:
        r1 = getattr(model, DMXR1_mapping[ii]).quantity
        r2 = getattr(model, DMXR2_mapping[ii]).quantity
        condition[DMX_mapping[ii]] = (r1.mjd, r2.mjd)
    return toas_selector.get_select_index(condition, toas["mjd_float"])


def dmxstats(model, toas, file=sys.stdout):
    """Print DMX statistics

    Based off dmxparse by P. Demorest (https://github.com/nanograv/tempo/tree/master/util/dmxparse)

    Parameters
    ----------
    model : pint.models.TimingModel
    toas : pint.toa.TOAs
    file : a file-like object (stream); defaults to the current sys.stdout
    """
    mjds = toas.get_mjds()
    freqs = toas.table["freq"]
    selected = np.zeros(len(toas), dtype=np.bool_)
    DMX_mapping = model.get_prefix_mapping("DMX_")
    select_idx = dmxselections(model, toas)
    for ii in DMX_mapping:
        if f"DMX_{ii:04d}" in select_idx:
            selection = select_idx[f"DMX_{ii:04d}"]
            selected[selection] = True
            print(
                "DMX_{:04d}: NTOAS={:5d}, MJDSpan={:14.4f}, FreqSpan={:8.3f}-{:8.3f}".format(
                    ii,
                    len(selection),
                    (mjds[selection].max() - mjds[selection.min()]),
                    freqs[selection].min() * u.MHz,
                    freqs[selection].max() * u.MHz,
                ),
                file=file,
            )
        else:
            print(
                "DMX_{:04d}: NTOAS={:5d}, MJDSpan={:14.4f}, FreqSpan={:8.3f}-{:8.3f}".format(
                    ii, 0, 0 * u.d, 0 * u.MHz, 0 * u.MHz
                ),
                file=file,
            )
    if not np.all(selected):
        print(f"{(1-selected).sum()} TOAs not selected in any DMX window", file=file)


def dmxparse(fitter, save=False):
    """Run dmxparse in python using PINT objects and results.

    Based off dmxparse by P. Demorest (https://github.com/nanograv/tempo/tree/master/util/dmxparse)

    Parameters
    ----------
    fitter
        PINT fitter used to get timing residuals, must have already run a fit
    save : bool or str or file-like object, optional
        If not False or None, saves output to specified file in the format of the TEMPO version.  If ``True``, assumes output file is ``dmxparse.out``

    Returns
    -------
    dict :

        ``dmxs`` : mean-subtraced dmx values

        ``dmx_verrs`` : dmx variance errors

        ``dmxeps`` : center mjds of the dmx bins

        ``r1s`` : lower mjd bounds on the dmx bins

        ``r2s`` : upper mjd bounds on the dmx bins

        ``bins`` : dmx bins

        ``mean_dmx`` : mean dmx value

        ``avg_dm_err`` : uncertainty in average dmx

    Raises
    ------
    RuntimeError
        If the model has no DMX parameters, or if there is a parsing problem

    """
    # We get the DMX values, errors, and mjds (same as in getting the DMX values for DMX v. time)
    # Get number of DMX epochs
    try:
        DMX_mapping = fitter.model.get_prefix_mapping("DMX_")
    except ValueError as e:
        raise RuntimeError("No DMX values in model!") from e
    dmx_epochs = [f"{x:04d}" for x in DMX_mapping.keys()]
    DMX_keys = list(DMX_mapping.values())
    DMXs = np.zeros(len(dmx_epochs))
    DMX_Errs = np.zeros(len(dmx_epochs))
    DMX_R1 = np.zeros(len(dmx_epochs))
    DMX_R2 = np.zeros(len(dmx_epochs))
    mask_idxs = np.zeros(len(dmx_epochs), dtype=np.bool_)
    # Get DMX values (will be in units of 10^-3 pc cm^-3)
    for ii, epoch in enumerate(dmx_epochs):
        DMXs[ii] = getattr(fitter.model, "DMX_{:}".format(epoch)).value
        mask_idxs[ii] = getattr(fitter.model, "DMX_{:}".format(epoch)).frozen
        DMX_Errs[ii] = getattr(fitter.model, "DMX_{:}".format(epoch)).uncertainty_value
        DMX_R1[ii] = getattr(fitter.model, "DMXR1_{:}".format(epoch)).value
        DMX_R2[ii] = getattr(fitter.model, "DMXR2_{:}".format(epoch)).value
    DMX_center_MJD = (DMX_R1 + DMX_R2) / 2
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
            sorted([f"DMX_{x}" for x in dmx_epochs])
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
        raise RuntimeError("Number of DMX entries do not match!")

    # Output the results'
    if save is not None and save:
        if isinstance(save, bool):
            save = "dmxparse.out"
        lines = [
            f"# Mean DMX value = {DMX_mean:+.6e} \n",
            f"# Uncertainty in average DM = {DMX_mean_err:.5e} \n",
            f"# Columns: DMXEP DMX_value DMX_var_err DMXR1 DMXR2 %s_bin \n",
        ]
        lines.extend(
            f"{DMX_center_MJD[k]:.4f} {DMXs[k] - DMX_mean:+.7e} {DMX_vErrs[k]:.3e} {DMX_R1[k]:.4f} {DMX_R2[k]:.4f} {DMX_keys[k]} \n"
            for k in range(len(dmx_epochs))
        )
        with open_or_use(save, mode="w") as dmxout:
            dmxout.writelines(lines)
            if isinstance(save, (str, Path)):
                log.debug(f"Wrote dmxparse output to '{save}'")
    # return the new mean subtracted values
    mean_sub_DMXs = DMXs - DMX_mean

    # Get units to multiply returned arrays by
    DMX_units = getattr(fitter.model, "DMX_{:}".format(dmx_epochs[0])).units
    DMXR_units = getattr(fitter.model, "DMXR1_{:}".format(dmx_epochs[0])).units

    return {
        "dmxs": mean_sub_DMXs * DMX_units,
        "dmx_verrs": DMX_vErrs * DMX_units,
        "dmxeps": DMX_center_MJD * DMXR_units,
        "r1s": DMX_R1 * DMXR_units,
        "r2s": DMX_R2 * DMXR_units,
        "bins": DMX_keys,
        "mean_dmx": DMX_mean * DMX_units,
        "avg_dm_err": DMX_mean_err * DMX_units,
    }


def get_prefix_timerange(model, prefixname):
    """Get time range for a prefix quantity like DMX or SWX

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
    prefixname : str
        Something like ``DMX_0001`` or ``SWX_0005``

    Returns
    -------
    tuple
        Each element is astropy.time.Time

    Example
    -------
    To match a range between SWX and DMX, you can do:

        >>> m.add_DMX_range(*(59077.33674631197, 59441.34020807681), index=1, frozen=False)

    Which sets ``DMX_0001`` to cover the same time range as ``SWX_0002``
    """
    prefix, index, indexnum = split_prefixed_name(prefixname)
    r1 = prefix.replace("_", "R1_") + index
    r2 = prefix.replace("_", "R2_") + index
    return getattr(model, r1).quantity, getattr(model, r2).quantity


def get_prefix_timeranges(model, prefixname):
    """Get all time ranges and indices for a prefix quantity like DMX or SWX

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
    prefixname : str
        Something like ``DMX`` or ``SWX`` (no trailing ``_``)

    Returns
    -------
    indices : np.ndarray
    starts : astropy.time.Time
    ends : astropy.time.Time

    """
    if prefixname.endswith("_"):
        prefixname = prefixname[:-1]
    prefix_mapping = model.get_prefix_mapping(f"{prefixname}_")
    r1 = np.zeros(len(prefix_mapping))
    r2 = np.zeros(len(prefix_mapping))
    indices = np.zeros(len(prefix_mapping), dtype=np.int32)
    for j, index in enumerate(prefix_mapping.keys()):
        if (
            getattr(model, f"{prefixname}R1_{index:04d}").quantity is not None
            and getattr(model, f"{prefixname}R2_{index:04d}").quantity is not None
        ):
            r1[j] = getattr(model, f"{prefixname}R1_{index:04d}").quantity.mjd
            r2[j] = getattr(model, f"{prefixname}R2_{index:04d}").quantity.mjd
            indices[j] = index
    return (
        indices,
        Time(r1, format="pulsar_mjd"),
        Time(r2, format="pulsar_mjd"),
    )


def find_prefix_bytime(model, prefixname, t):
    """Identify matching index(es) for a prefix parameter like DMX

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
    prefixname : str
        Something like ``DMX`` or ``SWX`` (no trailing ``_``)
    t : astropy.time.Time or float or astropy.units.Quantity
        If not :class:`astropy.time.Time`, then MJD is assumed

    Returns
    -------
    int or np.ndarray
        Index or indices that match
    """
    if not isinstance(t, Time):
        t = Time(t, format="pulsar_mjd")
    indices, r1, r2 = get_prefix_timeranges(model, prefixname)
    matches = np.where((t >= r1) & (t < r2))[0]
    if len(matches) == 1:
        matches = int(matches)
    return indices[matches]


def merge_dmx(model, index1, index2, value="mean", frozen=True):
    """Merge two DMX bins

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
    index1: int
    index2 : int
    value : str, optional
        One of "first", "second", "mean".  Determines value of new bin
    frozen : bool, optional

    Returns
    -------
    int
        New DMX index
    """
    assert value.lower() in ["first", "second", "mean"]
    tstart1, tend1 = get_prefix_timerange(model, f"DMX_{index1:04d}")
    tstart2, tend2 = get_prefix_timerange(model, f"DMX_{index2:04d}")
    tstart = min([tstart1, tstart2])
    tend = max([tend1, tend2])
    intervening_indices = find_prefix_bytime(model, "DMX", (tstart.mjd + tend.mjd) / 2)
    if len(np.setdiff1d(intervening_indices, [index1, index2])) > 0:
        for k in np.setdiff1d(intervening_indices, [index1, index2]):
            log.warning(
                f"Attempting to merge DMX_{index1:04d} and DMX_{index2:04d}, but DMX_{k:04d} is in between"
            )
    if value.lower() == "first":
        dmx = getattr(model, f"DMX_{index1:04d}").quantity
    elif value.lower == "second":
        dmx = getattr(model, f"DMX_{index2:04d}").quantity
    elif value.lower() == "mean":
        dmx = (
            getattr(model, f"DMX_{index1:04d}").quantity
            + getattr(model, f"DMX_{index2:04d}").quantity
        ) / 2
    # add the new one before we delete previous ones to make sure we have >=1 present
    newindex = model.add_DMX_range(tstart, tend, dmx=dmx, frozen=frozen)
    model.remove_DMX_range([index1, index2])
    return newindex


def split_dmx(model, time):
    """
    Split an existing DMX bin at the desired time

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
    time : astropy.time.Time

    Returns
    -------
    index : int
        Index of existing bin that was split
    newindex : int
        Index of new bin that was added

    """
    try:
        DMX_mapping = model.get_prefix_mapping("DMX_")
    except ValueError:
        raise RuntimeError("No DMX values in model!")
    dmx_epochs = [f"{x:04d}" for x in DMX_mapping.keys()]
    DMX_R1 = np.zeros(len(dmx_epochs))
    DMX_R2 = np.zeros(len(dmx_epochs))
    for ii, epoch in enumerate(dmx_epochs):
        DMX_R1[ii] = getattr(model, "DMXR1_{:}".format(epoch)).value
        DMX_R2[ii] = getattr(model, "DMXR2_{:}".format(epoch)).value
    ii = np.where((time.mjd > DMX_R1) & (time.mjd < DMX_R2))[0]
    if len(ii) == 0:
        raise ValueError(f"Time {time} not in any DMX bins")
    ii = ii[0]
    index = int(dmx_epochs[ii])
    t1 = DMX_R1[ii]
    t2 = DMX_R2[ii]
    print(f"{ii} {t1} {t2} {time}")
    getattr(model, f"DMXR2_{index:04d}").value = time.mjd
    newindex = model.add_DMX_range(
        time.mjd,
        t2,
        dmx=getattr(model, f"DMX_{index:04d}").quantity,
        frozen=getattr(model, f"DMX_{index:04d}").frozen,
    )
    return index, newindex


def split_swx(model, time):
    """
    Split an existing SWX bin at the desired time

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
    time : astropy.time.Time

    Returns
    -------
    index : int
        Index of existing bin that was split
    newindex : int
        Index of new bin that was added

    """
    try:
        SWX_mapping = model.get_prefix_mapping("SWX_")
    except ValueError:
        raise RuntimeError("No SWX values in model!")
    swx_epochs = [f"{x:04d}" for x in SWX_mapping.keys()]
    SWX_R1 = np.zeros(len(swx_epochs))
    SWX_R2 = np.zeros(len(swx_epochs))
    for ii, epoch in enumerate(swx_epochs):
        SWX_R1[ii] = getattr(model, "SWXR1_{:}".format(epoch)).value
        SWX_R2[ii] = getattr(model, "SWXR2_{:}".format(epoch)).value
    ii = np.where((time.mjd > SWX_R1) & (time.mjd < SWX_R2))[0]
    if len(ii) == 0:
        raise ValueError(f"Time {time} not in any SWX bins")
    ii = ii[0]
    index = int(swx_epochs[ii])
    t1 = SWX_R1[ii]
    t2 = SWX_R2[ii]
    print(f"{ii} {t1} {t2} {time}")
    getattr(model, f"SWXR2_{index:04d}").value = time.mjd
    newindex = model.add_swx_range(
        time.mjd,
        t2,
        swx=getattr(model, f"SWX_{index:04d}").quantity,
        frozen=getattr(model, f"SWX_{index:04d}").frozen,
    )
    return index, newindex


def wavex_setup(model, T_span, freqs=None, n_freqs=None, freeze_params=False):
    """
    Set-up a WaveX model based on either an array of user-provided frequencies or the wave number
    frequency calculation. Sine and Cosine amplitudes are initially set to zero

    User specifies T_span and either freqs or n_freqs. This function assumes that the timing model does not already
    have any WaveX components. See add_wavex_component() or add_wavex_components() to add WaveX components
    to an existing WaveX model.

    Parameters
    ----------

    model : pint.models.timing_model.TimingModel
    freqs : iterable of float or astropy.quantity.Quantity, None
        User inputed base frequencies
    n_freqs : int, None
        Number of wave frequencies to calculate using the equation: freq_n = 2 * pi * n / T_span
        Where n is the wave number, and T_span is the total time span of the toas in the fitter object
    T_span : float, astropy.quantity.Quantity
        Time span used to calculate nyquist frequency when using freqs
        Time span used to calculate WaveX frequencies when using n_freqs
        Usually to be set as the length of the timing baseline the model is being used for

    Returns
    -------

    indices : list
            Indices that have been assigned to new WaveX components
    """
    from pint.models.wavex import WaveX

    if (freqs is None) and (n_freqs is None):
        raise ValueError(
            "WaveX component base frequencies are not specified. "
            "Please input either freqs or n_freqs"
        )

    if (freqs is not None) and (n_freqs is not None):
        raise ValueError(
            "Both freqs and n_freqs are specified. Only one or the other should be used"
        )

    if n_freqs is not None and n_freqs <= 0:
        raise ValueError("Must use a non-zero number of wave frequencies")

    model.add_component(WaveX())
    if isinstance(T_span, u.quantity.Quantity):
        T_span.to(u.d)
    else:
        T_span *= u.d

    nyqist_freq = 1.0 / (2.0 * T_span)
    if freqs is not None:
        if isinstance(freqs, u.quantity.Quantity):
            freqs.to(u.d**-1)
        else:
            freqs *= u.d**-1
        if len(freqs) == 1:
            model.WXFREQ_0001.quantity = freqs
        else:
            freqs = np.array(freqs)
            freqs.sort()
            if min(np.diff(freqs)) < nyqist_freq:
                warnings.warn(
                    "Wave frequency spacing is finer than frequency resolution of data"
                )
            model.WXFREQ_0001.quantity = freqs[0]
            model.components["WaveX"].add_wavex_components(freqs[1:])

    if n_freqs is not None:
        if n_freqs == 1:
            wave_freq = 1 / T_span
            model.WXFREQ_0001.quantity = wave_freq
        else:
            wave_numbers = np.arange(1, n_freqs + 1)
            wave_freqs = wave_numbers / T_span
            model.WXFREQ_0001.quantity = wave_freqs[0]
            model.components["WaveX"].add_wavex_components(wave_freqs[1:])

    for p in model.params:
        if p.startswith("WXSIN") or p.startswith("WXCOS"):
            model[p].frozen = freeze_params

    return model.components["WaveX"].get_indices()


def dmwavex_setup(model, T_span, freqs=None, n_freqs=None, freeze_params=False):
    """
    Set-up a DMWaveX model based on either an array of user-provided frequencies or the wave number
    frequency calculation. Sine and Cosine amplitudes are initially set to zero

    User specifies T_span and either freqs or n_freqs. This function assumes that the timing model does not already
    have any DMWaveX components. See add_dmwavex_component() or add_dmwavex_components() to add components
    to an existing DMWaveX model.

    Parameters
    ----------

    model : pint.models.timing_model.TimingModel
    freqs : iterable of float or astropy.quantity.Quantity, None
        User inputed base frequencies
    n_freqs : int, None
        Number of wave frequencies to calculate using the equation: freq_n = 2 * pi * n / T_span
        Where n is the wave number, and T_span is the total time span of the toas in the fitter object
    T_span : float, astropy.quantity.Quantity
        Time span used to calculate nyquist frequency when using freqs
        Time span used to calculate WaveX frequencies when using n_freqs
        Usually to be set as the length of the timing baseline the model is being used for

    Returns
    -------

    indices : list
            Indices that have been assigned to new WaveX components
    """
    from pint.models.dmwavex import DMWaveX

    if (freqs is None) and (n_freqs is None):
        raise ValueError(
            "DMWaveX component base frequencies are not specified. "
            "Please input either freqs or n_freqs"
        )

    if (freqs is not None) and (n_freqs is not None):
        raise ValueError(
            "Both freqs and n_freqs are specified. Only one or the other should be used"
        )

    if n_freqs is not None and n_freqs <= 0:
        raise ValueError("Must use a non-zero number of wave frequencies")

    model.add_component(DMWaveX())
    if isinstance(T_span, u.quantity.Quantity):
        T_span.to(u.d)
    else:
        T_span *= u.d

    nyqist_freq = 1.0 / (2.0 * T_span)
    if freqs is not None:
        if isinstance(freqs, u.quantity.Quantity):
            freqs.to(u.d**-1)
        else:
            freqs *= u.d**-1
        if len(freqs) == 1:
            model.DMWXFREQ_0001.quantity = freqs
        else:
            freqs = np.array(freqs)
            freqs.sort()
            if min(np.diff(freqs)) < nyqist_freq:
                warnings.warn(
                    "DMWaveX frequency spacing is finer than frequency resolution of data"
                )
            model.DMWXFREQ_0001.quantity = freqs[0]
            model.components["DMWaveX"].add_dmwavex_components(freqs[1:])

    if n_freqs is not None:
        if n_freqs == 1:
            wave_freq = 1 / T_span
            model.DMWXFREQ_0001.quantity = wave_freq
        else:
            wave_numbers = np.arange(1, n_freqs + 1)
            wave_freqs = wave_numbers / T_span
            model.DMWXFREQ_0001.quantity = wave_freqs[0]
            model.components["DMWaveX"].add_dmwavex_components(wave_freqs[1:])

    for p in model.params:
        if p.startswith("DMWXSIN") or p.startswith("DMWXCOS"):
            model[p].frozen = freeze_params

    return model.components["DMWaveX"].get_indices()


def _translate_wave_freqs(om, k):
    """
    Use Wave model WAVEOM parameter to calculate a WaveX WXFREQ_ frequency parameter for wave number k

    Parameters
    ----------

    om : float or astropy.quantity.Quantity
        Base frequency of Wave model solution - parameter WAVEOM
        If float is given default units of 1/d assigned
    k : int
        wave number to use to calculate WaveX WXFREQ_ frequency parameter

    Returns
    -------

    WXFREQ_ quantity in units 1/d that can be used in WaveX model
    """
    if isinstance(om, u.quantity.Quantity):
        om.to(u.d**-1)
    else:
        om *= u.d**-1
    return (om * (k + 1)) / (2.0 * np.pi)


def _translate_wavex_freqs(wxfreq, k):
    """
    Use WaveX model WXFREQ_ parameters and wave number k to calculate the Wave model WAVEOM frequency parameter.

    Parameters
    ----------

    wxfreq : float or astropy.quantity.Quantity
        WaveX frequency from which the WAVEOM parameter will be calculated
        If float is given default units of 1/d assigned
    k : int
        wave number to use to calculate Wave WAVEOM parameter

    Returns
    -------

    WAVEOM quantity in units 1/d that can be used in Wave model
    """
    if isinstance(wxfreq, u.quantity.Quantity):
        wxfreq.to(u.d**-1)
    else:
        wxfreq *= u.d**-1
    if len(wxfreq) == 1:
        return (2.0 * np.pi * wxfreq) / (k + 1.0)
    wave_om = [((2.0 * np.pi * wxfreq[i]) / (k[i] + 1.0)) for i in range(len(wxfreq))]
    return (
        sum(wave_om) / len(wave_om)
        if np.allclose(wave_om, wave_om[0], atol=1e-3)
        else False
    )


def translate_wave_to_wavex(model):
    """
    Go from a Wave model to a WaveX model

    WaveX frequencies get calculated based on the Wave model WAVEOM parameter and the number of WAVE parameters.
        WXFREQ_000k = [WAVEOM * (k+1)] / [2 * pi]

    WaveX amplitudes are taken from the WAVE pair parameters

    Paramters
    ---------
    model : pint.models.timing_model.TimingModel
        TimingModel containing a Wave model to be converted to a WaveX model

    Returns
    -------
    New timing model with converted WaveX model included
    """
    from pint.models.wavex import WaveX

    new_model = deepcopy(model)
    wave_names = [
        f"WAVE{ii}" for ii in range(1, model.components["Wave"].num_wave_terms + 1)
    ]
    wave_terms = [getattr(model.components["Wave"], name) for name in wave_names]
    wave_om = model.components["Wave"].WAVE_OM.quantity
    wave_epoch = model.components["Wave"].WAVEEPOCH.quantity
    new_model.remove_component("Wave")
    new_model.add_component(WaveX())
    new_model.WXEPOCH.value = wave_epoch.value
    for k, wave_term in enumerate(wave_terms):
        wave_sin_amp, wave_cos_amp = wave_term.quantity
        wavex_freq = _translate_wave_freqs(wave_om, k)
        if k == 0:
            new_model.WXFREQ_0001.value = wavex_freq.value
            new_model.WXSIN_0001.value = -wave_sin_amp.value
            new_model.WXCOS_0001.value = -wave_cos_amp.value
        else:
            new_model.components["WaveX"].add_wavex_component(
                wavex_freq, wxsin=-wave_sin_amp, wxcos=-wave_cos_amp
            )
    return new_model


def get_wavex_freqs(model, index=None, quantity=False):
    """
    Return the WaveX frequencies for a timing model.

    If index is specified, returns the frequencies corresponding to the user-provided indices.
    If index isn't specified, returns all WaveX frequencies in timing model

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        Timing model from which to return WaveX frequencies
    index : float, int, list, np.ndarray, None
        Number or list/array of numbers corresponding to WaveX frequencies to return
    quantity : bool
        If set to True, returns a list of astropy.quanitity.Quantity rather than a list of prefixParameters

    Returns
    -------
    List of WXFREQ_ parameters
    """
    if index is None:
        freqs = model.components["WaveX"].get_prefix_mapping_component("WXFREQ_")
        if len(freqs) == 1:
            values = getattr(model.components["WaveX"], freqs.values())
        else:
            values = [
                getattr(model.components["WaveX"], param) for param in freqs.values()
            ]
    elif isinstance(index, (int, float, np.int64)):
        idx_rf = f"{int(index):04d}"
        values = getattr(model.components["WaveX"], f"WXFREQ_{idx_rf}")
    elif isinstance(index, (list, set, np.ndarray)):
        idx_rf = [f"{int(idx):04d}" for idx in index]
        values = [getattr(model.components["WaveX"], f"WXFREQ_{ind}") for ind in idx_rf]
    else:
        raise TypeError(
            f"index most be a float, int, set, list, array, or None - not {type(index)}"
        )
    if quantity:
        if len(values) == 1:
            values = [values[0].quantity]
        else:
            values = [v.quantity for v in values]
    return values


def get_wavex_amps(model, index=None, quantity=False):
    """
    Return the WaveX amplitudes for a timing model.

    If index is specified, returns the sine/cosine amplitudes corresponding to the user-provided indices.
    If index isn't specified, returns all WaveX sine/cosine amplitudes in timing model

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        Timing model from which to return WaveX frequencies
    index : float, int, list, np.ndarray, None
        Number or list/array of numbers corresponding to WaveX amplitudes to return
    quantity : bool
        If set to True, returns a list of tuples of astropy.quanitity.Quantity rather than a list of prefixParameters tuples

    Returns
    -------
    List of WXSIN_ and WXCOS_ parameters
    """
    if index is None:
        indices = (
            model.components["WaveX"].get_prefix_mapping_component("WXSIN_").keys()
        )
        if len(indices) == 1:
            values = getattr(
                model.components["WaveX"], f"WXSIN_{int(indices):04d}"
            ), getattr(model.components["WaveX"], f"WXCOS_{int(indices):04d}")
        else:
            values = [
                (
                    getattr(model.components["WaveX"], f"WXSIN_{int(idx):04d}"),
                    getattr(model.components["WaveX"], f"WXCOS_{int(idx):04d}"),
                )
                for idx in indices
            ]
    elif isinstance(index, (int, float, np.int64)):
        idx_rf = f"{int(index):04d}"
        values = getattr(model.components["WaveX"], f"WXSIN_{idx_rf}"), getattr(
            model.components["WaveX"], f"WXCOS_{idx_rf}"
        )
    elif isinstance(index, (list, set, np.ndarray)):
        idx_rf = [f"{int(idx):04d}" for idx in index]
        values = [
            (
                getattr(model.components["WaveX"], f"WXSIN_{ind}"),
                getattr(model.components["WaveX"], f"WXCOS_{ind}"),
            )
            for ind in idx_rf
        ]
    else:
        raise TypeError(
            f"index most be a float, int, set, list, array, or None - not {type(index)}"
        )
    if quantity:
        if isinstance(values, tuple):
            values = tuple(v.quantity for v in values)
        if isinstance(values, list):
            values = [(v[0].quantity, v[1].quantity) for v in values]
    return values


def translate_wavex_to_wave(model):
    """
    Go from a WaveX timing model to a Wave timing model.
    WARNING: Not every WaveX model can be appropriately translated into a Wave model. This is dependent on the user's choice of frequencies in the WaveX model.
    In order for a WaveX model to be able to be converted into a Wave model, every WaveX frequency must produce the same value of WAVEOM in the calculation:

    WAVEOM = [2 * pi * WXFREQ_000k] / (k + 1)
    Paramters
    ---------
    model : pint.models.timing_model.TimingModel
        TimingModel containing a WaveX model to be converted to a Wave model

    Returns
    -------
    New timing model with converted Wave model included
    """
    from pint.models.wave import Wave

    new_model = deepcopy(model)
    indices = model.components["WaveX"].get_indices()
    wxfreqs = get_wavex_freqs(model, indices, quantity=True)
    wave_om = _translate_wavex_freqs(wxfreqs, (indices - 1))
    if wave_om == False:
        raise ValueError(
            "This WaveX model cannot be properly translated into a Wave model due to the WaveX frequencies not producing a consistent WAVEOM value"
        )
    wave_amps = get_wavex_amps(model, index=indices, quantity=True)
    new_model.remove_component("WaveX")
    new_model.add_component(Wave())
    new_model.WAVEEPOCH.quantity = model.WXEPOCH.quantity
    new_model.WAVE_OM.quantity = wave_om
    new_model.WAVE1.quantity = tuple(w * -1.0 for w in wave_amps[0])
    if len(indices) > 1:
        for i in range(1, len(indices)):
            print(wave_amps[i])
            wave_amps[i] = tuple(w * -1.0 for w in wave_amps[i])
            new_model.components["Wave"].add_wave_component(
                wave_amps[i], index=indices[i]
            )
    return new_model


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
    wmean = (weights * arr).sum() / wtot if inputmean is None else float(inputmean)
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
    :math:`asini/c  e^4 \ll {\\rm timing precision} / \sqrt N_{\\rm TOA}`
    or :math:`A1 E^4 \ll TRES / \sqrt N_{\\rm TOA}`

    since the ELL1 model now includes terms up to O(E^3)

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
    lhs = A1 / const.c * E**4.0
    rhs = TRES / np.sqrt(NTOA)
    if outstring:
        s = (
            f"Checking applicability of ELL1 model -- \n"
            f"    Condition is asini/c * ecc**4 << timing precision / sqrt(# TOAs) to use ELL1\n"
            f"    asini/c * ecc**4    = {lhs.to(u.us):.3g} \n"
            f"    TRES / sqrt(# TOAs) = {rhs.to(u.us):.3g} \n"
        )
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
        return fdtrc(delta_dof, dof_2, F)
    elif dof_1 == dof_2:
        log.warning("Models have equal degrees of freedom, cannot perform F-test.")
        return np.nan
    else:
        log.warning(
            "Chi^2 for Model 2 is larger than Chi^2 for Model 1, cannot perform F-test."
        )
        return 1.0


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
            f"No proper motions available for {c}: returning coordinates unchanged"
        )
        return c

    if isinstance(c.frame, coords.builtin_frames.icrs.ICRS):
        return (
            coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra_cosdec,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                distance=distance,
                frame=coords.ICRS,
            )
            if hasattr(c, "pm_ra_cosdec")
            else coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                distance=distance,
                frame=coords.ICRS,
            )
        )
    elif isinstance(c.frame, coords.builtin_frames.galactic.Galactic):
        return coords.SkyCoord(
            l=c.l,
            b=c.b,
            pm_l_cosb=c.pm_l_cosb,
            pm_b=c.pm_b,
            obstime=c.obstime,
            distance=distance,
            frame=coords.Galactic,
        )
    elif isinstance(c.frame, pint.pulsar_ecliptic.PulsarEcliptic):
        return coords.SkyCoord(
            lon=c.lon,
            lat=c.lat,
            pm_lon_coslat=c.pm_lon_coslat,
            pm_lat=c.pm_lat,
            obstime=c.obstime,
            distance=distance,
            obliquity=c.obliquity,
            frame=pint.pulsar_ecliptic.PulsarEcliptic,
        )
    else:
        log.warning(
            f"Do not know coordinate frame for {c}: returning coordinates unchanged"
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
            f"No proper motions available for {c}: returning coordinates unchanged"
        )
        return c
    if isinstance(c.frame, coords.builtin_frames.icrs.ICRS):
        return (
            coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra_cosdec,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                frame=coords.ICRS,
            )
            if hasattr(c, "pm_ra_cosdec")
            else coords.SkyCoord(
                ra=c.ra,
                dec=c.dec,
                pm_ra_cosdec=c.pm_ra,
                pm_dec=c.pm_dec,
                obstime=c.obstime,
                frame=coords.ICRS,
            )
        )
    elif isinstance(c.frame, coords.builtin_frames.galactic.Galactic):
        return coords.SkyCoord(
            l=c.l,
            b=c.b,
            pm_l_cosb=c.pm_l_cosb,
            pm_b=c.pm_b,
            obstime=c.obstime,
            frame=coords.Galactic,
        )
    elif isinstance(c.frame, pint.pulsar_ecliptic.PulsarEcliptic):
        return coords.SkyCoord(
            lon=c.lon,
            lat=c.lat,
            pm_lon_coslat=c.pm_lon_coslat,
            pm_lat=c.pm_lat,
            obstime=c.obstime,
            obliquity=c.obliquity,
            frame=pint.pulsar_ecliptic.PulsarEcliptic,
        )
    else:
        log.warning(
            f"Do not know coordinate frame for {c}: returning coordinates unchanged"
        )
        return c


def info_string(prefix_string="# ", comment=None, detailed=False):
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
    detailed: bool, optional
        Include detailed version info on dependencies.

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

    info_dict = {
        "Created": f"{datetime.datetime.now().isoformat()}",
        "PINT_version": pint.__version__,
        "User": username,
        "Host": platform.node(),
        "OS": platform.platform(),
        "Python": sys.version,
    }

    if detailed:
        from numpy import __version__ as numpy_version
        from scipy import __version__ as scipy_version
        from astropy import __version__ as astropy_version
        from erfa import __version__ as erfa_version
        from jplephem import __version__ as jpleph_version
        from matplotlib import __version__ as matplotlib_version
        from loguru import __version__ as loguru_version
        from pint import __file__ as pint_file

        info_dict.update(
            {
                "endian": sys.byteorder,
                "numpy_version": numpy_version,
                "numpy_longdouble_precision": np.dtype(np.longdouble).name,
                "scipy_version": scipy_version,
                "astropy_version": astropy_version,
                "pyerfa_version": erfa_version,
                "jplephem_version": jpleph_version,
                "matplotlib_version": matplotlib_version,
                "loguru_version": loguru_version,
                "Python_prefix": sys.prefix,
                "PINT_file": pint_file,
            }
        )

        if "CONDA_PREFIX" in os.environ:
            conda_prefix = os.environ["CONDA_PREFIX"]
            info_dict.update(
                {
                    "Environment": "conda",
                    "conda_prefix": conda_prefix,
                }
            )
        elif "VIRTUAL_ENV" in os.environ:
            venv_prefix = os.environ["VIRTUAL_ENV"]
            info_dict.update(
                {
                    "Environment": "virtualenv",
                    "virtualenv_prefix": venv_prefix,
                }
            )

    s = "".join(f"{key}: {val}\n" for key, val in info_dict.items())
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


def colorize(text, fg_color=None, bg_color=None, attribute=None):
    """Colorizes a string (including unicode strings) for printing on the terminal

    For an example of usage, as well as a demonstration as to what the
    attributes and colors look like, check out :func:`~pint.utils.print_color_examples`

    Parameters
    ----------
    text : string
        The text to colorize. Can include unicode.
    fg_color : _type_, optional
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


def group_iterator(items):
    """An iterator to step over identical items in a :class:`numpy.ndarray`

    Example
    -------
    This will step over all of the observatories in the TOAs.
    For each iteration it gives the observatory name and the indices that correspond to it:

        >>> t = pint.toa.get_TOAs("grouptest.tim")
        >>> for o, i in group_iterator(t["obs"]):
        >>>     print(f"{o} {i}")

    """
    for item in np.unique(items):
        yield item, np.where(items == item)[0]


def compute_hash(filename):
    """Compute a unique hash of a file.

    This is designed to keep around to detect changes, not to be
    cryptographically robust. It uses the SHA256 algorithm, which
    is known to be vulnerable to a length-extension attack.

    Parameters
    ----------
    f : str or Path or file-like
        The source of input. If file-like, it should return ``bytes`` not ``str`` -
        that is, the file should be opened in binary mode.

    Returns
    -------
    bytes
        A cryptographic hash of the input.
    """
    h = hashlib.sha256()
    with open_or_use(filename, "rb") as f:
        # Reading in larger chunks saves looping without using
        # huge amounts of memory; and multiples of the hash
        # function block size are more efficient.
        blocks = 128
        while block := f.read(blocks * h.block_size):
            h.update(block)
    return h.digest()


def get_conjunction(coord, t0, precision="low", ecl="IERS2010"):
    """
    Find first time of Solar conjuction after t0 and approximate elongation at conjunction

    Offers a low-precision version (based on analytic expression of Solar longitude)
    Or a higher-precision version (based on interpolating :func:`astropy.coordinates.get_sun`)

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
    t0 : astropy.time.Time
    precision : str, optional
        "low" or "high" precision
    ecl : str, optional
        Obliquity for PulsarEcliptic coordinates

    Returns
    -------
    astropy.time.Time
        Time of conjunction
    astropy.units.Quantity
        Elongation at conjunction
    """

    assert precision.lower() in ["low", "high"]
    coord = coord.transform_to(pint.pulsar_ecliptic.PulsarEcliptic(ecl=ecl))

    # low precision version
    # use analytic form for Sun's ecliptic longitude
    # and interpolate
    tt = t0 + np.linspace(0, 365) * u.d
    # Allen's Astrophysical Quantities
    # Low precision solar coordinates (27.4.1)
    # number of days since J2000
    n = tt.jd - 2451545
    # mean longitude of Sun, corrected for abberation
    L = 280.460 * u.deg + 0.9854674 * u.deg * n
    # Mean anomaly
    g = 357.528 * u.deg + 0.9856003 * u.deg * n
    # Ecliptic longitude
    longitude = L + 1.915 * u.deg * np.sin(g) + 0.20 * u.deg * np.sin(2 * g)
    dlongitude = longitude - coord.lon
    dlongitude -= (dlongitude // (360 * u.deg)).max() * 360 * u.deg
    conjunction = Time(np.interp(0, dlongitude.value, tt.mjd), format="mjd")
    if precision.lower() == "low":
        return conjunction, coord.lat
    # do higher precision
    # use astropy solar coordinates
    # start with 10 days on either side of the low precision value
    tt = conjunction + np.linspace(-10, 10) * u.d
    csun = coords.get_sun(tt)
    # this seems to be needed in old astropy
    csun = coords.SkyCoord(ra=csun.ra, dec=csun.dec)
    elongation = csun.separation(coord)
    # get min value and interpolate with a quadratic fit
    j = np.where(elongation == elongation.min())[0][0]
    x = tt.mjd[j - 3 : j + 4]
    y = elongation.value[j - 3 : j + 4]
    f = np.polyfit(x, y, 2)
    conjunction = Time(-f[1] / 2 / f[0], format="mjd")
    csun = coords.get_sun(conjunction)
    # this seems to be needed in old astropy
    csun = coords.SkyCoord(ra=csun.ra, dec=csun.dec)

    return conjunction, csun.separation(coord)


def divide_times(t, t0, offset=0.5):
    """
    Divide input times into years relative to t0

    Years are centered around the requested offset value

    Parameters
    ----------
    t : astropy.time.Time
    t0 : astropy.time.Time
        Reference time
    offset : float, optional
        Offset value for division.  A value of 0.5 divides the results into intervals [-0.5,0.5].

    Returns
    -------
    np.ndarray
        Array of indices for division


    Example
    -------
    Divide into years around each conjunction

        >>> elongation = astropy.coordinates.get_sun(Time(t.get_mjds(), format="mjd")).separation(m.get_psr_coords())
        >>> t0 = get_conjunction(m.get_psr_coords(), m.PEPOCH.quantity, precision="high")[0]
        >>> indices = divide_times(Time(t.get_mjds(), format="mjd"), t0)
        >>> plt.clf()
        >>> for i in np.unique(indices):
                plt.plot(t.get_mjds()[indices == i], elongation[indices == i].value, ".")

    """
    dt = t - t0
    values = (dt.to(u.yr).value + offset) // 1
    return np.digitize(values, np.unique(values), right=True)


def convert_dispersion_measure(dm, dmconst=None):
    """Convert dispersion measure to a different value of the DM constant.

    Parameters
    ----------
    dm : astropy.units.Quantity
        DM measured according to the conventional value of the DM constant

    Returns
    -------
    dm : astropy.units.Quantity
        DM measured according to the value of the DM constant computed from the
        latest values of the physical constants
    dmconst : astropy.units.Quantity
        Value of the DM constant. Default value is computed from CODATA physical
        constants.
    Notes
    -----
    See https://nanograv-pint.readthedocs.io/en/latest/explanation.html#dispersion-measure
    for an explanation.
    """

    if dmconst is None:
        e = constants.e.si
        eps0 = constants.eps0.si
        c = constants.c.si
        me = constants.m_e.si
        dmconst = e**2 / (8 * np.pi**2 * c * eps0 * me)
    return (dm * pint.DMconst / dmconst).to(pint.dmu)


def parse_time(input, scale="tdb", precision=9):
    """Parse an :class:`astropy.time.Time` object from a range of input types

    Parameters
    ----------
    input : astropy.time.Time, astropy.units.Quantity, numpy.ndarray, float, int, str
        Value to parse
    scale : str, optional
        Scale of time for conversion
    precision : int, optional
        Precision for time

    Returns
    -------
    astropy.time.Time
    """
    if isinstance(input, Time):
        return input if input.scale == scale else getattr(input, scale)
    elif isinstance(input, u.Quantity):
        return Time(
            input.to(u.d), format="pulsar_mjd", scale=scale, precision=precision
        )
    elif isinstance(input, (np.ndarray, float, int)):
        return Time(input, format="pulsar_mjd", scale=scale, precision=precision)
    elif isinstance(input, str):
        return Time(input, format="pulsar_mjd_string", scale=scale, precision=precision)
    else:
        raise TypeError(f"Do not know how to parse times from {type(input)}")


def get_unit(parname):
    """Return the unit associated with a parameter

    Handles normal parameters, along with aliases and indexed parameters
    (e.g., `pint.models.parameter.prefixParameter`
    and `pint.models.parameter.maskParameter`) with an index beyond those currently
    initialized.

    This can be used without an existing :class:`~pint.models.TimingModel`.

    Parameters
    ----------
    name : str
        Name of PINT parameter or alias

    Returns
    -------
    astropy.u.Unit
    """
    # import in the function to avoid circular dependencies
    from pint.models.timing_model import AllComponents

    ac = AllComponents()
    return ac.param_to_unit(parname)


def normalize_designmatrix(M, params):
    """Normalize each row of the design matrix.

    This is used while computing the GLS chi2 and the GLS fitting step. The
    normalized and unnormalized design matrices Mn and M are related by
        M = Mn @ S
    where S is a diagonal matrix containing the norms. This normalization is
    OK because the GLS operations (fitting step, chi2 computation etc.) involve
    the form
        M @ (M.T @ N.inv() @ M).inv() @ M.T
    and it is easy to see that the above expression doesn't change if we replace
    M -> Mn.

    Different parameters can have different units and numerically vastly different
    design matrix entries. The normalization step forces the design matrix entries
    to have similar numericall values and hence improves the numerical precision of
    the matrix operations.
    """
    from pint.fitter import DegeneracyWarning

    norm = np.sqrt(np.sum(M**2, axis=0))

    bad_params = [params[i] for i in np.where(norm == 0)[0]]
    if len(bad_params) > 0 and params is not None:
        warn(
            f"Parameter degeneracy found in designmatrix! The offending parameters are {bad_params}.",
            DegeneracyWarning,
        )
    norm[norm == 0] = 1

    return M / norm, norm


def akaike_information_criterion(model, toas):
    """Compute the Akaike information criterion.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
        The timing model
    toas: pint.toas.TOAs
        TOAs

    Returns
    -------
    aic: float
        The Akaike information criterion
    """
    from pint.residuals import Residuals

    if not toas.is_wideband():
        k = (
            len(model.free_params)
            if "PhaseOffset" in model.components
            else len(model.free_params) + 1
        )
        lnL = Residuals(toas, model).lnlikelihood()
        return 2 * (k - lnL)
    else:
        raise NotImplementedError(
            "akaike_information_criterion is not yet implemented for wideband data."
        )


def sherman_morrison_dot(Ndiag, v, w, x, y):
    """
    Compute an inner product of the form
        (x| C^-1 |y)
    where
        C = N + w |v)(v| ,
    N is a diagonal matrix, and w is a positive real number,
    using the Sherman-Morrison identity
        C^-1 = N^-1 - ( w N^-1 |v)(v| N^-1 / (1 + w (v| N^-1 |v)) )

    Additionally,
        det[C] = det[N] * (1 + w (v| N^-1 |v)) )

    Paremeters
    ----------
    Ndiag: array-like
        Diagonal elements of the diagonal matrix N
    v: array-like
        A vector that represents a rank-1 update to N
    w: float
        Weight associated with the rank-1 update
    x: array-like
        Vector 1 for the inner product
    y: array-like
        Vector 2 for the inner product

    Returns
    -------
    result: float
        The inner product
    logdetC: float
        log-determinant of C
    """
    Ninv = 1 / Ndiag

    Ninv_v = Ninv * v
    denom = 1 + w * np.dot(v, Ninv_v)
    numer = w * np.dot(x, Ninv_v) * np.dot(y, Ninv_v)

    result = np.dot(x, Ninv * y) - numer / denom

    logdet_C = np.sum(np.log(Ndiag.to_value(u.s**2))) + np.log(
        denom.to_value(u.dimensionless_unscaled)
    )

    return result, logdet_C


def woodbury_dot(Ndiag, U, Phidiag, x, y):
    """
    Compute an inner product of the form
        (x| C^-1 |y)
    where
        C = N + U Phi U^T ,
    N and Phi are diagonal matrices, using the Woodbury
    identity
        C^-1 = N^-1 - N^-1 - N^-1 U Sigma^-1 U^T N^-1
    where
        Sigma = Phi^-1 + U^T N^-1 U

    Additionally,
        det[C] = det[N] * det[Phi] * det[Sigma]

    Paremeters
    ----------
    Ndiag: array-like
        Diagonal elements of the diagonal matrix N
    U: array-like
        A matrix that represents a rank-n update to N
    Phidiag: float
        Weights associated with the rank-n update
    x: array-like
        Vector 1 for the inner product
    y: array-like
        Vector 2 for the inner product

    Returns
    -------
    result: float
        The inner product
    logdetC: float
        log-determinant of C
    """

    x_Ninv_y = np.sum(x * y / Ndiag)
    x_Ninv_U = (x / Ndiag) @ U
    y_Ninv_U = (y / Ndiag) @ U
    Sigma = np.diag(1 / Phidiag) + (U.T / Ndiag) @ U
    Sigma_cf = cho_factor(Sigma)

    x_Cinv_y = x_Ninv_y - x_Ninv_U @ cho_solve(Sigma_cf, y_Ninv_U)

    logdet_N = np.sum(np.log(Ndiag))
    logdet_Phi = np.sum(np.log(Phidiag))
    _, logdet_Sigma = np.linalg.slogdet(Sigma)

    logdet_C = logdet_N + logdet_Phi + logdet_Sigma

    return x_Cinv_y, logdet_C


def _get_wx2pl_lnlike(model, component_name, ignore_fyr=True):
    from pint.models.noise_model import powerlaw
    from pint import DMconst

    assert component_name in ["WaveX", "DMWaveX"]
    prefix = "WX" if component_name == "WaveX" else "DMWX"

    idxs = np.array(model.components[component_name].get_indices())

    fs = np.array(
        [model[f"{prefix}FREQ_{idx:04d}"].quantity.to_value(u.Hz) for idx in idxs]
    )
    f0 = np.min(fs)
    fyr = (1 / u.year).to_value(u.Hz)

    assert np.allclose(
        np.diff(np.diff(fs)), 0
    ), "[DM]WaveX frequencies must be uniformly spaced."

    if ignore_fyr:
        year_mask = np.abs(((fs - fyr) / f0)) > 0.5

        idxs = idxs[year_mask]
        fs = np.array(
            [model[f"{prefix}FREQ_{idx:04d}"].quantity.to_value(u.Hz) for idx in idxs]
        )
        f0 = np.min(fs)

    scaling_factor = 1 if component_name == "WaveX" else DMconst / (1400 * u.MHz) ** 2

    a = np.array(
        [
            (scaling_factor * model[f"{prefix}SIN_{idx:04d}"].quantity).to_value(u.s)
            for idx in idxs
        ]
    )
    da = np.array(
        [
            (scaling_factor * model[f"{prefix}SIN_{idx:04d}"].uncertainty).to_value(u.s)
            for idx in idxs
        ]
    )
    b = np.array(
        [
            (scaling_factor * model[f"{prefix}COS_{idx:04d}"].quantity).to_value(u.s)
            for idx in idxs
        ]
    )
    db = np.array(
        [
            (scaling_factor * model[f"{prefix}COS_{idx:04d}"].uncertainty).to_value(u.s)
            for idx in idxs
        ]
    )

    def powl_model(params):
        """Get the powerlaw spectrum for the WaveX frequencies for a given
        set of parameters. This calls the powerlaw function used by `PLRedNoise`/`PLDMNoise`.
        """
        gamma, log10_A = params
        return (powerlaw(fs, A=10**log10_A, gamma=gamma) * f0) ** 0.5

    def mlnlike(params):
        """Negative of the likelihood function that acts on the
        `[DM]WaveX` amplitudes."""
        sigma = powl_model(params)
        return 0.5 * np.sum(
            (a**2 / (sigma**2 + da**2))
            + (b**2 / (sigma**2 + db**2))
            + np.log(sigma**2 + da**2)
            + np.log(sigma**2 + db**2)
        )

    return mlnlike


def plrednoise_from_wavex(model, ignore_fyr=True):
    """Convert a `WaveX` representation of red noise to a `PLRedNoise`
    representation. This is done by minimizing a likelihood function
    that acts on the `WaveX` amplitudes over the powerlaw spectral
    parameters.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
        The timing model with a `WaveX` component.
    ignore_fyr: bool
        Whether to ignore the frequency bin containinf 1 yr^-1
        while fitting for the spectral parameters.

    Returns
    -------
    pint.models.timing_model.TimingModel
        The timing model with a converted `PLRedNoise` component.
    """
    from pint.models.noise_model import PLRedNoise

    mlnlike = _get_wx2pl_lnlike(model, "WaveX", ignore_fyr=ignore_fyr)

    result = minimize(mlnlike, [4, -13], method="Nelder-Mead")
    if not result.success:
        raise ValueError("Log-likelihood maximization failed to converge.")

    gamma_val, log10_A_val = result.x

    hess = Hessian(mlnlike)
    gamma_err, log10_A_err = np.sqrt(
        np.diag(np.linalg.pinv(hess((gamma_val, log10_A_val))))
    )

    tnredc = len(model.components["WaveX"].get_indices())

    model1 = deepcopy(model)
    model1.remove_component("WaveX")
    model1.add_component(PLRedNoise())
    model1.TNREDAMP.value = log10_A_val
    model1.TNREDGAM.value = gamma_val
    model1.TNREDC.value = tnredc
    model1.TNREDAMP.uncertainty_value = log10_A_err
    model1.TNREDGAM.uncertainty_value = gamma_err

    return model1


def pldmnoise_from_dmwavex(model, ignore_fyr=False):
    """Convert a `DMWaveX` representation of red noise to a `PLDMNoise`
    representation. This is done by minimizing a likelihood function
    that acts on the `DMWaveX` amplitudes over the powerlaw spectral
    parameters.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
        The timing model with a `DMWaveX` component.

    Returns
    -------
    pint.models.timing_model.TimingModel
        The timing model with a converted `PLDMNoise` component.
    """
    from pint.models.noise_model import PLDMNoise

    mlnlike = _get_wx2pl_lnlike(model, "DMWaveX", ignore_fyr=ignore_fyr)

    result = minimize(mlnlike, [4, -13], method="Nelder-Mead")
    if not result.success:
        raise ValueError("Log-likelihood maximization failed to converge.")

    gamma_val, log10_A_val = result.x

    hess = Hessian(mlnlike)

    H = hess((gamma_val, log10_A_val))
    assert np.all(np.linalg.eigvals(H) > 0), "The Hessian is not positive definite!"

    Hinv = np.linalg.pinv(H)
    assert np.all(
        np.linalg.eigvals(Hinv) > 0
    ), "The inverse Hessian is not positive definite!"

    gamma_err, log10_A_err = np.sqrt(np.diag(Hinv))

    tndmc = len(model.components["DMWaveX"].get_indices())

    model1 = deepcopy(model)
    model1.remove_component("DMWaveX")
    model1.add_component(PLDMNoise())
    model1.TNDMAMP.value = log10_A_val
    model1.TNDMGAM.value = gamma_val
    model1.TNDMC.value = tndmc
    model1.TNDMAMP.uncertainty_value = log10_A_err
    model1.TNDMGAM.uncertainty_value = gamma_err

    return model1


def find_optimal_nharms(model, toas, component, nharms_max=45):
    """Find the optimal number of harmonics for `WaveX`/`DMWaveX` using the Akaike Information
    Criterion.

    Parameters
    ----------
    model: `pint.models.timing_model.TimingModel`
        The timing model. Should not already contain `WaveX`/`DMWaveX` or `PLRedNoise`/`PLDMNoise`.
    toas: `pint.toa.TOAs`
        Input TOAs
    component: str
        Component name; "WaveX" or "DMWaveX"
    nharms_max: int
        Maximum number of harmonics

    Returns
    -------
    nharms_opt: int
        Optimal number of harmonics
    aics: ndarray
        Array of normalized AIC values.
    """
    from pint.fitter import Fitter

    assert component in ["WaveX", "DMWaveX"]
    assert (
        component not in model.components
    ), f"{component} is already included in the model."
    assert (
        "PLRedNoise" not in model.components and "PLDMNoise" not in model.components
    ), "PLRedNoise/PLDMNoise cannot be included in the model."

    model1 = deepcopy(model)

    ftr = Fitter.auto(toas, model1, downhill=False)
    ftr.fit_toas(maxiter=5)
    aics = [akaike_information_criterion(model1, toas)]
    model1 = ftr.model

    T_span = toas.get_mjds().max() - toas.get_mjds().min()
    setup_component = wavex_setup if component == "WaveX" else dmwavex_setup
    setup_component(model1, T_span, n_freqs=1, freeze_params=False)

    for _ in range(nharms_max):
        ftr = Fitter.auto(toas, model1, downhill=False)
        ftr.fit_toas(maxiter=5)
        aics.append(akaike_information_criterion(ftr.model, toas))

        model1 = ftr.model
        if component == "WaveX":
            model1.components[component].add_wavex_component(
                (len(model1.components[component].get_indices()) + 1) / T_span,
                frozen=False,
            )
        else:
            model1.components[component].add_dmwavex_component(
                (len(model1.components[component].get_indices()) + 1) / T_span,
                frozen=False,
            )

    assert all(np.isfinite(aics)), "Infs/NaNs found in AICs!"

    return np.argmin(aics), np.array(aics) - np.min(aics)
