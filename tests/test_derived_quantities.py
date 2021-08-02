"""Test basic functionality of the :module:`pint.derived_quantities`."""
import os
from itertools import product
from tempfile import NamedTemporaryFile

import astropy.constants as c
import astropy.units as u
import numpy as np
import pytest
import scipy.stats
from astropy.time import Time
from hypothesis import assume, example, given
from hypothesis.extra.numpy import array_shapes, arrays, scalar_dtypes
from hypothesis.strategies import (
    composite,
    floats,
    integers,
    just,
    one_of,
    sampled_from,
    slices,
)
from numdifftools import Derivative
from numpy.testing import assert_allclose, assert_array_equal
from pinttestdata import datadir

import pint
from pint.derived_quantities import (
    a1sini,
    companion_mass,
    mass_funct,
    mass_funct2,
    pulsar_age,
    pulsar_B,
    pulsar_B_lightcyl,
    pulsar_edot,
    pulsar_mass,
    omdot,
    pbdot,
    gamma,
    omdot_to_mtot,
)


def test_mass_function():

    # Mass function
    # RHS of Eqn. 8.34 in Lorimer & Kramer (2008)
    # this should be 4* pi**2 * x**3 / (G * Pb**2)
    # in appropriate units
    assert np.isclose(
        mass_funct(1.0 * u.d, 2.0 * pint.ls), 0.008589595519643776 * u.solMass
    )


def test_other_mass_function():

    # Mass function, second form
    # LHS of Eqn. 8.34 in Lorimer & Kramer (2008)
    # this should be (Mc * sin(inc))**3 / (Mp + Mc)**2
    assert np.isclose(
        mass_funct2(1.4 * u.solMass, 0.2 * u.solMass, 60.0 * u.deg),
        0.0020297470401197783 * u.solMass,
    )


def test_characteristic_age():
    # Characteristic age
    assert np.isclose(
        pulsar_age(0.033 * u.Hz, -2.0e-15 * u.Hz / u.s), 261426.72446573884 * u.yr
    )


def test_Edot():
    # Edot
    assert np.isclose(
        pulsar_edot(0.033 * u.Hz, -2.0e-15 * u.Hz / u.s),
        2.6055755618875905e30 * u.erg / u.s,
    )


def test_Bfield():
    # B
    assert np.isclose(
        pulsar_B(0.033 * u.Hz, -2.0e-15 * u.Hz / u.s), 238722891596281.66 * u.G
    )


def test_Blc():
    # B_lc
    assert np.isclose(
        pulsar_B_lightcyl(0.033 * u.Hz, -2.0e-15 * u.Hz / u.s),
        0.07774704753236616 * u.G,
    )


@given(
    floats(min_value=0.5, max_value=3),
    floats(min_value=0.01, max_value=10),
    floats(min_value=0.04, max_value=1000),
    floats(min_value=0.1, max_value=90),
)
def test_companion_mass(Mpsr, Mc, Pb, incl):
    """
    test companion mass calculations for a range of values
    """
    Mc = Mc * u.Msun
    Mpsr = Mpsr * u.Msun
    Pb = Pb * u.day
    incl = incl * u.deg

    Mtot = Mc + Mpsr
    # full semi-major axis
    a = (c.G * Mtot * (Pb / (2 * np.pi)) ** 2) ** (1.0 / 3)
    # pulsar semi-major axis
    apsr = (Mc / Mtot) * a
    # projected
    x = (apsr * np.sin(incl)).to(pint.ls)
    # computed companion mass
    assert np.isclose(companion_mass(Pb, x, mp=Mpsr, i=incl), Mc)


@given(
    arrays(float, 10, elements=floats(0.5, 3)),
    arrays(float, 10, elements=floats(0.01, 10)),
    arrays(float, 10, elements=floats(0.04, 1000)),
    arrays(float, 10, elements=floats(0.1, 90)),
)
def test_companion_mass_array(Mpsr, Mc, Pb, incl):
    """
    test companion mass calculations for a range of values given np.ndarray inputs
    """
    Mc = Mc * u.Msun
    Mpsr = Mpsr * u.Msun
    Pb = Pb * u.day
    incl = incl * u.deg

    Mtot = Mc + Mpsr
    # full semi-major axis
    a = (c.G * Mtot * (Pb / (2 * np.pi)) ** 2) ** (1.0 / 3)
    # pulsar semi-major axis
    apsr = (Mc / Mtot) * a
    # projected
    x = (apsr * np.sin(incl)).to(pint.ls)
    # computed companion mass
    assert np.allclose(companion_mass(Pb, x, mp=Mpsr, i=incl), Mc)


@given(
    floats(min_value=0.5, max_value=3),
    floats(min_value=0.01, max_value=10),
    floats(min_value=0.04, max_value=1000),
    floats(min_value=0.1, max_value=90),
)
def test_pulsar_mass(Mpsr, Mc, Pb, incl):
    """
    test pulsar mass calculations for a range of values
    """
    Mc = Mc * u.Msun
    Mpsr = Mpsr * u.Msun
    Pb = Pb * u.day
    incl = incl * u.deg

    Mtot = Mc + Mpsr
    # full semi-major axis
    a = (c.G * Mtot * (Pb / (2 * np.pi)) ** 2) ** (1.0 / 3)
    # pulsar semi-major axis
    apsr = (Mc / Mtot) * a
    # projected
    x = (apsr * np.sin(incl)).to(pint.ls)
    # computed pulsar mass
    assert np.isclose(pulsar_mass(Pb, x, Mc, i=incl), Mpsr)


@given(
    arrays(float, 10, elements=floats(0.5, 3)),
    arrays(float, 10, elements=floats(0.01, 10)),
    arrays(float, 10, elements=floats(0.04, 1000)),
    arrays(float, 10, elements=floats(0.1, 90)),
)
def test_pulsar_mass_array(Mpsr, Mc, Pb, incl):
    """
    test pulsar mass calculations for a range of values given np.ndarray inputs
    """
    Mc = Mc * u.Msun
    Mpsr = Mpsr * u.Msun
    Pb = Pb * u.day
    incl = incl * u.deg

    Mtot = Mc + Mpsr
    # full semi-major axis
    a = (c.G * Mtot * (Pb / (2 * np.pi)) ** 2) ** (1.0 / 3)
    # pulsar semi-major axis
    apsr = (Mc / Mtot) * a
    # projected
    x = (apsr * np.sin(incl)).to(pint.ls)
    # computed pulsar mass
    assert np.allclose(pulsar_mass(Pb, x, Mc, i=incl), Mpsr)


def test_omdot():
    Mp = 1.3381 * u.Msun
    Mc = 1.2489 * u.Msun
    Pb = 0.10225156248 * u.d
    e = 0.0877775 * u.dimensionless_unscaled
    # for the Double Pulsar
    # https://arxiv.org/pdf/astro-ph/0609417.pdf
    # but recalculated based on the values above
    assert np.isclose(omdot(Mp, Mc, Pb, e), 16.8991396 * u.deg / u.yr)


@given(
    floats(min_value=1.0, max_value=3.0),
    floats(min_value=0.01, max_value=10),
    floats(min_value=0.04, max_value=1000),
    floats(min_value=0.0001, max_value=0.99),
)
def test_omdot_to_mtot(Mp, Mc, Pb, e):
    Mp = Mp * u.Msun
    Mc = Mc * u.Msun
    Pb = Pb * u.d
    e = e * u.dimensionless_unscaled
    # compute the total mass
    Mtot = Mp + Mc
    # compute the omdot
    omdot_computed = omdot(Mp, Mc, Pb, e)
    # compute the Mtot from that omdot and compare
    Mtot_computed = omdot_to_mtot(omdot_computed, Pb, e)
    assert np.allclose(Mtot, Mtot_computed)


def test_gamma():
    Mp = 1.3381 * u.Msun
    Mc = 1.2489 * u.Msun
    Pb = 0.10225156248 * u.d
    e = 0.0877775 * u.dimensionless_unscaled
    # for the Double Pulsar
    # https://arxiv.org/pdf/astro-ph/0609417.pdf
    # but recalculated based on the values above
    assert np.isclose(gamma(Mp, Mc, Pb, e), 0.38402 * u.ms)


def test_pbdot():
    Mp = 1.3381 * u.Msun
    Mc = 1.2489 * u.Msun
    Pb = 0.10225156248 * u.d
    e = 0.0877775 * u.dimensionless_unscaled
    # for the Double Pulsar
    # https://arxiv.org/pdf/astro-ph/0609417.pdf
    # but recalculated based on the values above
    assert np.isclose(pbdot(Mp, Mc, Pb, e), -1.24777223e-12 * u.s / u.s)


@given(
    floats(min_value=1.0, max_value=3.0),
    floats(min_value=0.01, max_value=10),
    floats(min_value=0.04, max_value=1000),
    floats(min_value=0.1, max_value=90),
)
def test_a1sini_Mc(Mp, Mc, Pb, i):
    """test a1sini by looking for consistency with companion mass calculation."""
    Mp = Mp * u.Msun
    Mc = Mc * u.Msun
    Pb = Pb * u.d
    i = i * u.deg
    x = a1sini(Mp, Mc, Pb, i=i)
    assert np.isclose(Mc, companion_mass(Pb, x, i=i, mp=Mp))


@given(
    floats(min_value=1.0, max_value=3.0),
    floats(min_value=0.01, max_value=10),
    floats(min_value=0.04, max_value=1000),
    floats(min_value=0.1, max_value=90),
)
def test_a1sini_Mp(Mp, Mc, Pb, i):
    """test a1sini by looking for consistency with pulsar mass calculation."""
    Mp = Mp * u.Msun
    Mc = Mc * u.Msun
    Pb = Pb * u.d
    i = i * u.deg
    x = a1sini(Mp, Mc, Pb, i=i)
    assert np.isclose(Mp, pulsar_mass(Pb, x, Mc, i))


@given(
    arrays(float, 10, elements=floats(0.5, 3)),
    arrays(float, 10, elements=floats(0.01, 10)),
    arrays(float, 10, elements=floats(0.04, 1000)),
    arrays(float, 10, elements=floats(0.1, 90)),
)
def test_a1sini_Mc_array(Mp, Mc, Pb, i):
    """test a1sini by looking for consistency with companion mass calculation with array input."""
    Mp = Mp * u.Msun
    Mc = Mc * u.Msun
    Pb = Pb * u.d
    i = i * u.deg
    x = a1sini(Mp, Mc, Pb, i=i)
    assert np.allclose(Mc, companion_mass(Pb, x, i=i, mp=Mp))


@given(
    arrays(float, 10, elements=floats(0.5, 3)),
    arrays(float, 10, elements=floats(0.01, 10)),
    arrays(float, 10, elements=floats(0.04, 1000)),
    arrays(float, 10, elements=floats(0.1, 90)),
)
def test_a1sini_Mp_array(Mp, Mc, Pb, i):
    """test a1sini by looking for consistency with pulsar mass calculation with array input."""
    Mp = Mp * u.Msun
    Mc = Mc * u.Msun
    Pb = Pb * u.d
    i = i * u.deg
    x = a1sini(Mp, Mc, Pb, i=i)
    assert np.allclose(Mp, pulsar_mass(Pb, x, Mc, i))
