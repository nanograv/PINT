# Test for phase.py

import pytest
import numpy as np
import astropy.units as u
from pint.phase import Phase


# modified from @mhvk's test_phase_class.py
def assert_equal(first, second):
    assert type(first) is type(second)
    if type(first) == int:
        assert np.all(first == second)
    else:
        # if floating point, cannot check for equality, check for closeness
        # .all() for vector implementation (array returned)
        assert np.isclose(first, second, rtol=1e-9, atol=0).all()
    assert first.unit == second.unit == u.dimensionless_unscaled


class TestPhaseInit:
    @pytest.mark.parametrize(
        "inti, fraci, intf, fracf",
        (
            (0, 0, 0, 0),  # zero case
            (2, 0.3, 2, 0.3),  # regular case
            (1, 0.7, 2, -0.3),  # frac out of range, less than 1
            (-4, 0.5, -3, -0.5),  # edge case, 0.5
            (4, -0.5, 4, -0.5),  # edge case, -0.5
            (5, 1.4, 6, 0.4),  # frac greater than 1
            (1.2, 0, 1, 0.2),  # frac in int
            (1.2, 0.2, 1, 0.4),  # frac in int and frac
            (2 * u.Unit(""), 0.3 * u.Unit(""), 2, 0.3),  # initialized w/ u
        ),
    )
    def test_init_scalar(self, inti, fraci, intf, fracf):
        phase = Phase(inti, fraci)
        assert isinstance(phase, Phase)
        assert_equal(phase.int, u.Quantity(intf))
        assert_equal(phase.frac, u.Quantity(fracf))

    def test_init_array(self):
        phase = Phase([0, 2, -4, 1.2, 5], [0, 0.3, 0.5, 0, 1.4])
        assert isinstance(phase, Phase)
        assert_equal(phase.int, u.Quantity([0, 2, -3, 1, 6]))
        assert_equal(phase.frac, u.Quantity([0, 0.3, -0.5, 0.2, 0.4]))

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
        assert_equal(phasesum.int, u.Quantity(sumi))
        assert_equal(phasesum.frac, u.Quantity(sumf))
        assert np.isfinite(phasesum.quantity.value)
        assert phasesum.quantity.value == phasesum.value

    def test_commutative_scalar_addition(self):
        phase1 = Phase(2, 0.5)
        phase2 = Phase(1, 0.3)
        sum1 = phase1 + phase2
        sum2 = phase2 + phase1
        assert_equal(sum1.int, sum2.int)
        assert_equal(sum1.frac, sum2.frac)

    def test_associative_scalar_addition(self):
        phase1 = Phase(2, 0.5)
        phase2 = Phase(1, 0.3)
        phase3 = Phase(3, -0.1)
        sum1 = phase1 + (phase2 + phase3)
        sum2 = (phase1 + phase2) + phase3
        assert_equal(sum1.int, sum2.int)
        assert_equal(sum1.frac, sum2.frac)

    def test_scalar_negation(self):
        phase1 = Phase(2, 0.3)
        phase2 = -phase1
        sum = phase1 + phase2
        assert_equal(sum.int, u.Quantity(0))
        assert_equal(sum.frac, u.Quantity(0))
        phase01 = -Phase(0, 0)
        assert_equal(phase01.int, u.Quantity(0))
        assert_equal(phase01.frac, u.Quantity(0))

    def test_scalar_multiplication(self):
        phase = Phase(2, 0.1)
        product1 = phase * 0
        assert isinstance(product1, Phase)
        assert_equal(product1.int, u.Quantity(0))
        assert_equal(product1.frac, u.Quantity(0))
        product2 = phase * 1
        assert_equal(product2.int, phase.int)
        assert_equal(product2.frac, phase.frac)
        product3 = 2 * phase
        assert_equal(product3.int, u.Quantity(4))
        assert_equal(product3.frac, u.Quantity(0.2))

    def test_precision(self):
        phase = Phase(1e5, 0.1)
        phase2 = phase + Phase(0, 1e-9)
        assert_equal(phase2.int, u.Quantity(1e5))
        assert_equal(phase2.frac, u.Quantity(0.100000001))


class TestVectorArithmeticFunc:
    def test_vector_addition(self):
        phase1 = Phase([0, 2, 2, 2], [0, 0.3, 0.3, 0])
        phase2 = Phase([0, 1, 1, 1], [0, 0.1, 0.2, -0.5])
        phasesum = phase1 + phase2
        assert isinstance(phasesum, Phase)
        assert_equal(phasesum.int, u.Quantity([0, 3, 4, 3]))
        assert_equal(phasesum.frac, u.Quantity([0, 0.4, -0.5, -0.5]))

    def test_commutative_vector_addition(self):
        phase1 = Phase([0, 2, 2, 2], [0, 0.3, 0.3, 0])
        phase2 = Phase([0, 1, 1, 1], [0, 0.1, 0.2, -0.5])
        sum1 = phase1 + phase2
        sum2 = phase2 + phase1
        assert_equal(sum1.int, sum2.int)
        assert_equal(sum1.frac, sum2.frac)

    def test_associative_vector_addition(self):
        phase1 = Phase([0, 2, 2, 2], [0, 0.3, 0.3, 0])
        phase2 = Phase([0, 1, 1, 1], [0, 0.1, 0.2, -0.5])
        phase3 = Phase([1, 5, 2, 3], [0.2, 0.4, -0.3, 0.3])
        sum1 = phase1 + (phase2 + phase3)
        sum2 = (phase1 + phase2) + phase3
        assert_equal(sum1.int, sum2.int)
        assert_equal(sum1.frac, sum2.frac)

    def test_vector_addition_with_scalar(self):
        vecphase = Phase([0, 2, 2, 2], [0, 0.3, 0.3, 0])
        scalarphase = Phase(1, 0.1)
        sum1 = vecphase + scalarphase
        assert isinstance(sum1, Phase)
        assert_equal(sum1.int, u.Quantity([1, 3, 3, 3]))
        assert_equal(sum1.frac, u.Quantity([0.1, 0.4, 0.4, 0.1]))
        # check commutivity
        sum2 = scalarphase + vecphase
        assert isinstance(sum2, Phase)
        assert_equal(sum1.int, sum2.int)
        assert_equal(sum1.frac, sum2.frac)

    def test_vector_negation(self):
        phase1 = Phase([1, -2, -3, 4], [0.1, -0.3, 0.4, -0.2])
        phase2 = -phase1
        sum = phase1 + phase2
        assert_equal(sum.int, u.Quantity(0))
        assert_equal(sum.frac, u.Quantity(0))
        phase01 = -Phase([0, 0], [0, 0])
        assert_equal(phase01.int, u.Quantity(0))
        assert_equal(phase01.frac, u.Quantity(0))

    def test_vector_multiplication(self):
        phase = Phase([2, 1, -3], [0.1, -0.4, 0.2])
        product1 = phase * 0
        assert isinstance(product1, Phase)
        assert_equal(product1.int, u.Quantity([0, 0, 0]))
        assert_equal(product1.frac, u.Quantity([0, 0, 0]))
        product2 = phase * 1
        assert_equal(product2.int, phase.int)
        assert_equal(product2.frac, phase.frac)
        product3 = phase * 2
        assert_equal(product3.int, u.Quantity([4, 1, -6]))
        assert_equal(product3.frac, u.Quantity([0.2, 0.2, 0.4]))
