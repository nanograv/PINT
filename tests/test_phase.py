# Test for phase.py

import pytest
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
from astropy.coordinates import Angle
from pint.phase import Phase, FractionalPhase


def assert_equal(one, other):
    """Check matching type, matching phase1,2 and that phase1 is integer."""
    assert type(one) is type(other)
    assert np.all(one == other)
    if isinstance(one, Phase):
        assert one.unit == other.unit == u.cycle
        assert one.imaginary == other.imaginary
        assert np.all(one.view(np.ndarray)['int'] % 1. == 0)
    elif hasattr(one, 'unit'):
        assert one.unit == other.unit


class TestPhaseInit:
    @pytest.mark.parametrize(
        "inti, fraci, intf, fracf",
        (
            (0, 0, 0, 0),  # zero case
            (2, 0.3, 2, 0.3),  # regular case
            (1, 0.75, 2, -0.25),  # frac out of range, less than 1
            (-5, 0.5, -4, -0.5),  # edge case, odd
            (4, -0.5, 4, -0.5),  # edge case, -0.5
            (5, 1.375, 6, 0.375),  # frac greater than 1
            (1.25, 0, 1, 0.25),  # frac in int
            (1.25, 0.125, 1, 0.375),  # frac in int and frac
            (2 * u.Unit(""), 0.375 * u.Unit(""), 2, 0.375),  # initialized w/ u
        ),
    )
    def test_init_scalar(self, inti, fraci, intf, fracf):
        phase = Phase(inti, fraci)
        assert isinstance(phase, Phase)
        assert_array_equal(phase.int, Angle(intf, u.cycle))
        assert_array_equal(phase.frac, FractionalPhase(fracf))

    def test_init_array(self):
        phase = Phase([0, 2, -3, 1.25, 5], [0, 0.375, 0.5, 0, 1.375])
        assert isinstance(phase, Phase)
        assert_array_equal(phase.int, Angle([0, 2, -2, 1, 6], u.cycle))
        assert_array_equal(phase.frac, FractionalPhase([0, 0.375, -0.5, 0.25, 0.375]))

    def test_init_bad_unit(self):
        # bad units on int
        try:
            Phase(2 * u.Unit("m"), 0.3)
        except u.core.UnitConversionError:
            pass
        else:
            print("Exception not thrown")
            raise u.core.UnitConversionError
        # bad units on frac
        try:
            Phase(2, 0.3 * u.Unit("m"))
        except u.core.UnitConversionError:
            pass
        else:
            print("Exception not thrown")
            raise u.core.UnitConversionError
        # bad units on int and frac
        try:
            Phase(2 * u.Unit("m"), 0.3 * u.Unit("m"))
        except u.core.UnitConversionError:
            pass
        else:
            print("Exception not thrown")
            raise u.core.UnitConversionError


class TestScalarArithmeticFunc:
    @pytest.mark.parametrize(
        "ii1, ff1, ii2, ff2, sumi, sumf",
        (
            (0, 0, 0, 0, 0, 0),  # zero case
            (2, 0.3, 1, 0.1, 3, 0.4),  # regular case
            (2, 0.3, 1, 0.2, 4, -0.5),  # edge case, adds to 0.5
            (2, 0, 1, -0.5, 3, -0.5),  # edge case, adds to -0.5
        ),
    )
    def test_scalar_addition(self, ii1, ff1, ii2, ff2, sumi, sumf):
        phase1 = Phase(ii1, ff1)
        phase2 = Phase(ii2, ff2)
        phasesum = phase1 + phase2
        assert isinstance(phasesum, Phase)
        assert_equal(phasesum, Phase(sumi, sumf))

    def test_commutative_scalar_addition(self):
        phase1 = Phase(2, 0.5)
        phase2 = Phase(1, 0.3)
        sum1 = phase1 + phase2
        sum2 = phase2 + phase1
        assert_equal(sum1, sum2)

    def test_associative_scalar_addition(self):
        # Note that this can only be true for number for which no floating point
        # errors are made.
        phase1 = Phase(2, 0.5)
        phase2 = Phase(1, 0.375)
        phase3 = Phase(3, -0.125)
        sum1 = phase1 + (phase2 + phase3)
        sum2 = (phase1 + phase2) + phase3
        assert_equal(sum1, sum2)

    def test_scalar_negation(self):
        phase1 = Phase(2, 0.3)
        phase2 = -phase1
        sum = phase1 + phase2
        assert_equal(sum, Phase(0))
        phase01 = -Phase(0, 0)
        assert_equal(phase01, Phase(0))

    def test_scalar_multiplication(self):
        phase = Phase(2, 0.1)
        product1 = phase * 0
        assert isinstance(product1, Phase)
        assert_equal(product1, Phase(0))
        product2 = phase * 1
        assert_equal(product2, phase)
        product3 = phase * 2
        assert_equal(product3, Phase(4, 0.2))

    def test_precision(self):
        phase = Phase(1e5, 0.1)
        phase2 = phase + Phase(0, 1e-9)
        assert_equal(phase2, Phase(1e5, 0.100000001))


class TestVectorArithmeticFunc:
    def test_vector_addition(self):
        phase1 = Phase([0, 2, 2, 2], [0, 0.3, 0.3, 0])
        phase2 = Phase([0, 1, 1, 1], [0, 0.1, 0.2, -0.5])
        phasesum = phase1 + phase2
        assert_equal(phasesum, Phase([0, 3, 4, 3], [0, 0.4, -0.5, -0.5]))

    def test_commutative_vector_addition(self):
        phase1 = Phase([0, 2, 2, 2], [0, 0.3, 0.3, 0])
        phase2 = Phase([0, 1, 1, 1], [0, 0.1, 0.2, -0.5])
        sum1 = phase1 + phase2
        sum2 = phase2 + phase1
        assert_equal(sum1, sum2)

    def test_associative_vector_addition(self):
        # Note that this can only be true for number for which no floating point
        # errors are made.
        phase1 = Phase([0, 2, 2, 2], [0, 0.375, 0.375, 0])
        phase2 = Phase([0, 1, 1, 1], [0, 0.125, 0.25, -0.5])
        phase3 = Phase([1, 5, 2, 3], [0.25, 0.375, -0.375, 0.375])
        sum1 = phase1 + (phase2 + phase3)
        sum2 = (phase1 + phase2) + phase3
        assert_equal(sum1, sum2)

    def test_vector_addition_with_scalar(self):
        vecphase = Phase([0, 2, 2, 2], [0, 0.3, 0.3, 0])
        scalarphase = Phase(1, 0.1)
        sum1 = vecphase + scalarphase
        assert isinstance(sum1, Phase)
        assert_equal(sum1, Phase([1, 3, 3, 3], [0.1, 0.4, 0.4, 0.1]))
        # check commutivity
        sum2 = scalarphase + vecphase
        assert isinstance(sum2, Phase)
        assert_equal(sum1, sum2)

    def test_vector_negation(self):
        phase1 = Phase([1, -2, -3, 4], [0.1, -0.3, 0.4, -0.2])
        phase2 = -phase1
        sum = phase1 + phase2
        assert_equal(sum, Phase(0))
        phase01 = -Phase([0, 0], [0, 0])
        assert_equal(phase01, Phase(0))

    def test_vector_multiplication(self):
        phase = Phase([2, 1, -3], [0.1, -0.375, 0.2])
        product1 = phase * 0
        assert isinstance(product1, Phase)
        assert_equal(product1, Phase(0))
        product2 = phase * 1
        assert_equal(product2, phase)
        product3 = phase * 2
        assert_equal(product3, Phase([4, 1, -6], [0.2, 0.25, 0.4]))
