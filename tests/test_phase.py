# Test for phase.py
import operator

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

    def test_init_with_phase(self):
        phase = Phase(1., 0.125)
        phase2 = Phase(phase)
        assert_equal(phase2, phase)
        assert phase2 is not phase
        assert not np.may_share_memory(phase2, phase)
        phase3 = Phase(phase, copy=False)
        assert phase3 is phase
        phase4 = Phase(phase, 0., copy=False)
        assert phase4 is not phase
        assert_equal(phase4, phase)
        phase5 = Phase(0., phase)
        assert phase5 is not phase
        assert_equal(phase5, phase)
        phase6 = Phase(phase, phase)
        assert_equal(phase6, Phase(2., 0.25))

    def test_init_with_subclass(self):
        class MyPhase(Phase):
            pass

        my_phase = MyPhase(1., 0.25)
        assert type(my_phase) is MyPhase
        phase2 = Phase(my_phase)
        assert type(phase2) is Phase
        phase3 = Phase(my_phase, subok=True)
        assert type(phase3) is MyPhase
        assert phase3 is not my_phase
        assert not np.may_share_memory(phase3, my_phase)
        phase4 = Phase(my_phase, copy=False)
        assert type(phase4) is Phase
        assert np.may_share_memory(phase4, my_phase)
        phase5 = Phase(my_phase, copy=False, subok=True)
        assert phase5 is my_phase
        phase6 = Phase(my_phase, 0., copy=False, subok=True)
        assert type(phase6) is MyPhase
        assert not np.may_share_memory(phase6, my_phase)
        phase7 = Phase(my_phase, phase2, copy=False, subok=True)
        assert type(phase7) is MyPhase
        phase8 = Phase(phase2, my_phase, copy=False, subok=True)
        assert type(phase8) is MyPhase

    def test_init_complex(self):
        phase = Phase(1j)
        assert isinstance(phase, Phase)
        assert phase.imaginary
        assert_equal(phase.int, Angle(1j, u.cycle))
        assert_equal(phase.frac, Angle(0j, u.cycle))
        assert_equal(phase.cycle, Angle(1j, u.cycle))
        assert '1j cycle' in repr(phase)

        phase2 = Phase(1 + 0j)
        assert isinstance(phase2, Phase)
        assert not phase2.imaginary
        assert_equal(phase2, Phase(1))
        assert '1j cycle' not in repr(phase2)

        with pytest.raises(ValueError):
            Phase(1., 0.0001j)


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


class PhaseSetup:
    def setup(self):
        self.phase1 = Angle(np.array([1000., 1001., 999., 1005, 1006.]),
                            u.cycle)[:, np.newaxis]
        self.phase2 = Angle(2.**(-53) * np.array([1, -1., 1., -1.])
                            + np.array([-0.5, 0., 0., 0.5]), u.cycle)
        self.phase = Phase(self.phase1, self.phase2)
        self.delta = Phase(0., self.phase2)
        self.im_phase = Phase(self.phase1 * 1j, self.phase2 * 1j)


class TestPhase(PhaseSetup):
    def test_basics(self):
        assert isinstance(self.phase, Phase)
        assert np.all(self.phase.int % (1. * u.cycle) == 0)
        cycle = self.phase1 + self.phase2
        assert_equal(self.phase.cycle, cycle)
        assert_equal(self.phase.int, Angle(self.phase1))
        assert_equal(self.phase.frac, FractionalPhase(self.phase2))

    @pytest.mark.parametrize('in1,in2', ((1.1111111, 0),
                                         (1.5, 0.111),
                                         (0.11111111, 1),
                                         (1.*u.deg, 0),
                                         (1.*u.cycle, 1.*u.deg)))
    def test_phase1_always_integer(self, in1, in2):
        phase = Phase(in1, in2)
        assert phase.int % (1. * u.cycle) == 0
        expected = u.Quantity(in1 + in2, u.cycle).value
        assert (phase.int + phase.frac).value == expected
        assert phase.value == expected

    def test_astype(self):
        float64 = self.phase.astype('f8')
        assert_equal(float64, self.phase.cycle)
        float32 = self.phase.astype('f4')
        assert_equal(float32, self.phase.cycle.astype('f4'))
        copy = self.phase.astype(self.phase.dtype)
        assert copy is not self.phase
        assert not np.may_share_memory(copy, self.phase)
        assert_equal(copy, self.phase)
        same = self.phase.astype(self.phase.dtype, copy=False)
        assert same is self.phase

    def test_conversion(self):
        degrees = self.phase.to(u.degree)
        assert_equal(degrees, Angle(self.phase1 + self.phase2, u.deg))

    @pytest.mark.parametrize('item', (0, (0, 1), (slice(2), 1)))
    def test_selection(self, item):
        phase1 = self.phase[item]
        expected_cycle = self.phase.cycle[item]
        assert phase1.shape == expected_cycle.shape
        assert_equal(phase1.cycle, expected_cycle)
        phase2 = self.im_phase[item]
        expected_cycle2 = self.im_phase.cycle[item]
        assert phase2.shape == expected_cycle2.shape
        assert_equal(phase2.cycle, expected_cycle2)

    def test_equality(self):
        phase2 = self.phase[:, 1:2]
        eq = self.phase == phase2
        expected = [False, True, False, False]
        assert np.all(eq == expected)

        im_phase2 = self.im_phase[:, 1:2]
        eq = self.im_phase == im_phase2
        assert np.all(eq == expected)

        eq_real_imag = phase2 == im_phase2
        assert eq_real_imag.shape == phase2.shape
        assert not np.any(eq_real_imag)

    def test_addition(self):
        add = self.phase + self.phase
        assert_equal(add, Phase(2. * self.phase1, 2. * self.phase2))
        t = self.phase1 + self.phase
        add2 = self.phase2 + t
        assert_equal(add2, add)
        t = self.phase + self.phase1.to(u.degree)
        add3 = t + self.phase2.to(u.degree)
        assert_equal(add3, add)
        add4 = self.phase + 1. * u.cycle
        assert_equal(add4, Phase(self.phase1 + 1 * u.cycle, self.phase2))
        add5 = 360. * u.deg + self.phase
        assert_equal(add5, add4)

        add6 = self.phase + self.im_phase
        assert_equal(add6, self.phase.cycle + self.im_phase.cycle)

    def test_subtraction(self):
        double = Phase(self.phase1 * 2., self.phase2 * 2.)
        sub = double - self.phase
        assert_equal(sub, self.phase)
        t = self.phase2 * 2. - self.phase
        sub2 = self.phase1 * 2. + t
        assert_equal(sub2, sub)
        t = double - self.phase1.to(u.degree)
        sub3 = t - self.phase2
        assert_equal(sub3, sub)
        sub4 = self.phase - 1. * u.cycle
        assert_equal(sub4, Phase(self.phase1 - 1 * u.cycle, self.phase2))

        sub6 = self.phase - self.im_phase
        assert_equal(sub6, self.phase.cycle - self.im_phase.cycle)

    def test_inplace_addition_subtraction(self):
        add = self.phase.copy()
        link = add
        add += self.phase
        assert add is link
        assert_equal(add, Phase(2. * self.phase1, 2. * self.phase2))

        result = np.subtract(add, self.phase, out=add)
        assert result is add
        assert_equal(result, self.phase)

        # Quantity output should work.
        out = self.phase.cycle
        out += self.phase
        assert_equal(out, 2 * self.phase.cycle)

        result2 = np.subtract(out, self.phase, out=out)
        assert result2 is out
        assert_equal(out, self.phase.cycle)

        with pytest.raises(TypeError):  # array output is not OK.
            np.add(out, self.phase, out=out.value)

        with pytest.raises(TypeError):
            out += self.im_phase

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison(self, op):
        result = op(self.phase, self.phase[0])
        assert_equal(result, op((self.phase - self.phase[0]).cycle, 0.))
        # Also for small differences.
        result = op(self.phase, self.phase[:, 1:2])
        assert_equal(result, op((self.phase - self.phase[:, 1:2]).cycle, 0.))

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison_quantity(self, op):
        ref = 1005. * u.cy
        result = op(self.phase, ref.to(u.deg))
        assert_equal(result, op((self.phase - ref).cycle, 0.))

    def test_comparison_invalid_quantity(self):
        # Older astropy uses UnitConversionError
        with pytest.raises((TypeError, u.UnitConversionError)):
            self.phase > 1. * u.m

        with pytest.raises((TypeError, u.UnitConversionError)):
            self.phase <= 1. * u.m

        assert (self.phase == 1. * u.m) is False
        assert (self.phase != 1. * u.m) is True

    def test_comparison_with_out(self):
        out = np.empty(self.phase.shape, bool)
        result = np.less(self.phase, self.phase[0], out=out)
        assert result is out
        assert_equal(result, np.less((self.phase - self.phase[0]).cycle, 0.))
        with pytest.raises(TypeError):
            np.less(self.phase, self.phase[0], out=self.phase)

    def test_negative(self):
        neg = -self.phase
        assert_equal(neg, Phase(-self.phase1, -self.phase2))
        neg2 = -self.im_phase
        assert_equal(neg2, Phase(-1j*self.phase1, -1j*self.phase2))

    def test_negative_with_out(self):
        out = 0 * self.phase
        result = np.negative(self.phase, out=out)
        assert result is out
        assert_equal(result, Phase(-self.phase1, -self.phase2))
        # Also with regular Angle input.
        out2 = Phase(np.zeros_like(self.phase1))
        result2 = np.negative(self.phase1, out=out2)
        assert result2 is out2
        assert_equal(result2, Phase(-self.phase1))
        # And for imaginary output
        result = np.negative(self.im_phase, out=out)
        assert result is out
        assert_equal(result, Phase(-1j*self.phase1, -1j*self.phase2))

    def test_absolute(self):
        check1 = abs(-self.phase)
        assert_equal(check1, self.phase)
        check2 = abs(self.im_phase)
        assert_equal(check2, self.phase)
        check3 = abs(-self.im_phase)
        assert_equal(check3, self.phase)

    def test_absolute_with_out(self):
        out = 0 * self.phase
        result = np.absolute(-self.phase, out=out)
        assert result is out
        assert_equal(result, self.phase)
        # Also with regular Angle input.
        out2 = Phase(np.zeros_like(self.phase1))
        result2 = np.absolute(-self.phase1, out=out2)
        assert result2 is out2
        assert_equal(result2, Phase(self.phase1))
        # And with imaginary phase
        out3 = 0 * self.im_phase
        result3 = np.absolute(-self.im_phase, out=out3)
        assert result3 is out3
        assert_equal(result3, self.phase)

    def test_rint(self):
        out = np.rint(self.phase)
        assert_equal(out, self.phase.int)
        out2 = np.rint(self.im_phase)
        assert_equal(out2, self.im_phase.int)

    def test_rint_with_out(self):
        out = 0 * self.phase
        result = np.rint(self.phase, out=out)
        assert result is out
        assert_equal(result, Phase(self.phase.int))
        out2 = np.empty(self.phase.shape) * u.cycle
        result2 = np.rint(self.phase, out=out2)
        assert result2 is out2
        expected = u.Quantity(self.phase.int)
        assert_equal(result2, expected)

    def test_unitless_multiplication(self):
        mul = self.phase * 2
        assert_equal(mul, Phase(self.phase1 * 2, self.phase2 * 2))
        mul2 = self.phase * (2. * u.dimensionless_unscaled)
        assert_equal(mul2, mul)
        mul3 = self.phase * 2. * u.one
        assert_equal(mul3, mul)
        mul4 = 2. * self.phase
        assert_equal(mul4, mul)
        mul5 = self.phase * np.full(self.phase.shape, 2.)
        assert_equal(mul5, mul)

    def test_unitless_division(self):
        div = self.phase / 0.5
        assert_equal(div, Phase(self.phase1 * 2, self.phase2 * 2))
        div2 = self.phase / (0.5 * u.dimensionless_unscaled)
        assert_equal(div2, div)
        div3 = self.phase / 0.5 / u.one
        assert_equal(div3, div)
        div4 = self.phase / np.full(self.phase.shape, 0.5)
        assert_equal(div4, div)

    def test_inplace_unitless_multiplication_division(self):
        out = self.phase.copy()
        link = out
        out *= 2
        assert out is link
        assert_equal(out, Phase(self.phase1 * 2, self.phase2 * 2))
        out /= 2
        assert out is link
        assert_equal(out, self.phase)
        out = np.multiply(self.phase, 2, out=out)
        assert out is link
        assert_equal(out, Phase(self.phase1 * 2, self.phase2 * 2))
        out = np.divide(self.phase, 0.5, out=out)
        assert out is link
        assert_equal(out, Phase(self.phase1 * 2, self.phase2 * 2))
        # Also for input angles.
        out = np.multiply(self.phase.cycle, 2, out=out)
        assert out is link
        assert_equal(out, Phase(2 * self.phase.cycle))
        with pytest.raises(u.UnitsError):
            out *= 2 * u.m
        with pytest.raises(u.UnitsError):
            np.multiply(self.phase.cycle, 2 * u.m, out=out)

    def test_unitfull_multiplication(self):
        mul = self.phase * (2 * u.Hz)
        assert_equal(mul, u.Quantity(self.phase.cycle * 2 * u.Hz))
        mul2 = self.phase * 2. * u.Hz
        assert_equal(mul2, mul)
        mul3 = 2. * u.Hz * self.phase
        assert_equal(mul3, mul)
        mul4 = 2. * u.Hz * np.ones(self.phase.shape)
        link = mul4
        mul4 *= self.phase
        assert mul4 is link
        assert_equal(mul4, mul)

    def test_unitfull_division(self):
        div = self.phase / (0.5 * u.s)
        assert_equal(div, u.Quantity(self.phase.cycle * 2 / u.s))
        div2 = self.phase / 0.5 / u.s
        assert_equal(div2, div)
        div3 = 0.5 * u.s / self.phase
        assert_equal(div3, 1. / div)
        div4 = 0.5 * u.s * np.ones(self.phase.shape)
        link = div4
        div4 /= self.phase
        assert div4 is link
        assert_equal(div4, 1. / div)

    def test_imaginary_scalings(self):
        mul = self.phase * 1j
        expected = Phase(self.phase1 * 1j, self.phase2 * 1j)
        assert_equal(mul, expected)
        mul2 = self.phase * 0.125j
        expected2 = expected * 0.125
        assert_equal(mul2, expected2)
        div = self.phase / 8j
        expected3 = -expected2
        assert_equal(div, expected3)
        mul4 = self.phase * (1+1j)
        expected4 = self.phase.cycle * (1+1j)
        assert_equal(mul4, expected4)

    def test_floor_division_mod(self):
        fd = self.phase // (1. * u.cycle)
        fd_exp = self.phase.int.copy()
        fd_exp[self.phase.frac < 0] -= 1 * u.cycle
        fd_exp = fd_exp / u.cycle
        assert_equal(fd, fd_exp)
        mod = self.phase % (1. * u.cycle)
        mod_exp = Phase(np.where(self.phase.frac >= 0., 0., 1.),
                        self.phase.frac)
        assert_equal(mod, mod_exp)
        exp_cycle = Angle(self.phase.frac, copy=True)
        exp_cycle[exp_cycle < 0.] += 1. * u.cycle
        assert_equal(mod.cycle, exp_cycle)
        dm = divmod(self.phase, 1. * u.cycle)
        assert_equal(dm[0], fd_exp)
        assert_equal(dm[1], mod_exp)
        #
        fd2 = self.phase // (360. * u.degree)
        assert_equal(fd2, fd_exp)
        mod2 = self.phase % (360 * u.degree)
        assert_equal(mod2, mod_exp)
        dm2 = divmod(self.phase, 360 * u.degree)
        assert_equal(dm2[0], fd_exp)
        assert_equal(dm2[1], mod_exp)
        #
        fd3 = self.phase // (240. * u.hourangle)
        fd3_exp = fd_exp // 10
        assert_equal(fd3, fd3_exp)
        mod3 = self.phase % (240. * u.hourangle)
        mod_int_exp = self.phase.int % (10 * u.cy)
        mod_int_exp[0][self.phase.frac[0] < 0] += 10. * u.cy
        mod3_exp = Phase(mod_int_exp, self.phase.frac)
        assert_equal(mod3, mod3_exp)
        dm3 = divmod(self.phase, 240. * u.hourangle)
        assert_equal(dm3[0], fd3_exp)
        assert_equal(dm3[1], mod3_exp)

        with pytest.raises(u.UnitsError):
            np.mod(self.phase, 1. * u.m)

    def test_floor_division_mod_with_out(self):
        out1 = np.empty(self.phase.shape) * u.dimensionless_unscaled
        fd = np.floor_divide(self.phase, 1. * u.cycle, out=out1)
        assert fd is out1
        fd_exp = self.phase // (1. * u.cycle)  # checked above.
        assert_equal(fd, fd_exp)
        out2 = 0. * self.phase
        mod = np.mod(self.phase, 1. * u.cycle, out=out2)
        assert mod is out2
        mod_exp = Phase(np.where(self.phase.frac >= 0., 0., 1.),
                        self.phase.frac)
        assert_equal(mod, mod_exp)
        out1 *= 0
        out2 *= 0
        dm = np.divmod(self.phase, 1. * u.cycle, out=(out1, out2))
        assert dm[0] is out1
        assert dm[1] is out2
        assert_equal(dm[0], fd_exp)
        assert_equal(dm[1], mod_exp)
        with pytest.raises(TypeError):
            np.floor_divide(self.phase, 1. * u.cycle, out=out2)
        with pytest.raises(TypeError):
            np.divmod(self.phase, 1. * u.cycle, out=(out2, out1))

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_min(self, axis):
        m = self.phase.min(axis=axis)
        index = (slice(None) if axis == 1 else self.phase1.argmin(),
                 slice(None) if axis == 0 else self.phase2.argmin())
        assert_equal(m, self.phase[index])

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_max(self, axis):
        m = self.phase.max(axis=axis)
        index = (slice(None) if axis == 1 else self.phase1.argmax(),
                 slice(None) if axis == 0 else self.phase2.argmax())
        assert_equal(m, self.phase[index])

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_ptp(self, axis):
        ptp = self.phase.ptp(axis)
        assert_equal(ptp, self.phase.max(axis) - self.phase.min(axis))

    @pytest.mark.parametrize('axis', (0, 1))
    def test_sort(self, axis):
        sort = self.phase.sort(axis=axis)
        if axis == 1:
            index = ()
        else:
            index = self.phase1.ravel().argsort()
        assert_equal(sort, self.phase[index])

    @pytest.mark.parametrize('ufunc', (np.sin, np.cos, np.tan))
    def test_trig(self, ufunc):
        d = np.arange(-177, 180, 10) * u.degree
        cycle = 1e10 * u.cycle
        expected = ufunc(d)
        assert not np.isclose(ufunc(cycle + d), expected,
                              atol=1e-14, rtol=1.e-14).any()
        phase = Phase(cycle, d)
        assert np.isclose(ufunc(phase), expected, rtol=1e-14,
                          atol=1e-14).all()

    def test_trig_with_out(self):
        out = np.zeros(self.phase.shape) * u.dimensionless_unscaled
        result = np.sin(self.phase, out=out)
        assert out is result
        assert np.all(np.sin(self.phase2) == result)
        # Need quantity to store output.
        with pytest.raises(TypeError):
            np.sin(self.phase, out=self.phase)
        # Also bail if one got there because of out.
        with pytest.raises(TypeError):
            np.sin(self.phase2, out=self.phase)

    def test_exp(self):
        in_ = self.phase * 1j
        out = np.exp(in_)
        expected = np.exp(self.phase.frac.to_value(u.radian) * 1j) * u.one
        assert_equal(out, expected)

        out *= 0
        result = np.exp(in_, out=out)
        assert result is out
        assert_equal(result, expected)

        with pytest.raises(u.UnitsError):
            np.exp(self.phase)

    def test_spacing(self):
        out = np.spacing(self.phase)
        expected = np.spacing(self.phase.frac)
        assert_equal(out, expected)

    def test_spacing_with_out(self):
        out = u.Quantity(np.empty(self.phase.shape), u.cycle)
        result = np.spacing(self.phase, out=out)
        assert result is out
        assert_equal(out, u.Quantity(np.spacing(self.phase.frac)))
        out2 = 0 * self.phase
        result2 = np.spacing(self.phase, out=out2)
        assert result2 is out2
        assert_equal(result2, Phase(result))

    def test_isnan(self):
        expected = np.zeros(self.phase.shape)
        assert_equal(np.isnan(self.phase), expected)
        # For older astropy, we set input to nan rather than Phase directly,
        # since setting of nan exposes a Quantity bug.
        phase2 = self.phase2.copy()
        phase2[1] = np.nan
        phase = Phase(self.phase1, phase2)

        expected[:, 1] = True
        assert_equal(np.isnan(phase), expected)
        trial = Phase(np.nan)
        assert np.isnan(trial)


class TestPhaseString(PhaseSetup):
    def test_to_string_basic(self):
        s = self.phase.to_string()
        assert np.all(s == [
            ['999.5000000000000001', '999.9999999999999999',
             '1000.0000000000000001', '1000.4999999999999999'],
            ['1000.5000000000000001', '1000.9999999999999999',
             '1001.0000000000000001', '1001.4999999999999999'],
            ['998.5000000000000001', '998.9999999999999999',
             '999.0000000000000001', '999.4999999999999999'],
            ['1004.5000000000000001', '1004.9999999999999999',
             '1005.0000000000000001', '1005.4999999999999999'],
            ['1005.5000000000000001', '1005.9999999999999999',
             '1006.0000000000000001', '1006.4999999999999999']],)

    def test_format_string_basic(self):
        s = self.phase.to_string()
        for ph, expected in zip(self.phase.ravel(), s.ravel()):
            assert '{:.16f}'.format(ph) == expected

    def test_to_string_precision(self):
        s = self.phase.to_string(precision=5)
        assert np.all(s == [
            ['999.50000', '1000.00000', '1000.00000', '1000.50000'],
            ['1000.50000', '1001.00000', '1001.00000', '1001.50000'],
            ['998.50000', '999.00000', '999.00000', '999.50000'],
            ['1004.50000', '1005.00000', '1005.00000', '1005.50000'],
            ['1005.50000', '1006.00000', '1006.00000', '1006.50000']])
        for ph, expected in zip(self.phase.ravel(), s.ravel()):
            assert '{:.5f}'.format(ph) == expected

    def test_to_string_alwayssign(self):
        ph = Phase([[-10], [20]], [-0.4, 0.4])
        s = ph.to_string(alwayssign=True)
        assert np.all(s == [['-10.4', '-9.6'],
                            ['+19.6', '+20.4']])
        for ph, expected in zip(ph.ravel(), s.ravel()):
            assert '{:+.1f}'.format(ph) == expected

    @pytest.mark.parametrize('count,frac,expected', (
        (1., 0., '1.0'), (1., 0.1, '1.1'), (-10, 0.05, '-9.95'),
        (-10, -0.05, '-10.05'), (-10., -0.025, '-10.025'),
        (10, 0.05, '10.05'), (1000., 0., '1000.0'),
        (-10., 0.4, '-9.6')))
    def test_to_string_corner_cases(self, count, frac, expected):
        ph = Phase(count, frac)
        s = ph.to_string()
        assert s == expected

    def test_from_string_basic(self):
        p = Phase.from_string('9876543210.0123456789')
        assert p == Phase(9876543210, .0123456789)
        p = Phase.from_string('9876543210.0123456789j')
        assert p == Phase(9876543210j, .0123456789j)

    def test_from_string_with_exponents(self):
        p = Phase.from_string('9876543210.0123456789e-01')
        assert p == Phase(987654321, .00123456789)
        p = Phase.from_string('9876543210.0123456789D-01')
        assert p == Phase(987654321, .00123456789)
        # Check that we avoid round-off errors (in fractional phase).
        assert p != Phase(9876543210 * 1e-1, .0123456789 * 1e-1)
        p = Phase.from_string('9876543210.0123456789e-01j')
        assert p == Phase(987654321j, .00123456789j)
        p = Phase.from_string('9876543210.0123456789e1j')
        assert p == Phase(98765432100j, .123456789j)
        p = Phase.from_string('9876543210.0123456789e-12j')
        assert p == Phase(0j, .0098765432100123456789j)
        # Really not suited for large exponents...
        p = Phase.from_string('9876543210.0123456789e12j')
        assert p == Phase(9876543210012345678900j, 0j)

    @pytest.mark.parametrize('imag', (True, False))
    @pytest.mark.parametrize('alwayssign', (True, False))
    @pytest.mark.parametrize('precision', (None, 16))
    def test_to_from_string_roundtrip(self, precision, alwayssign, imag):
        p_in = self.phase * 1j if imag else self.phase
        s = p_in.to_string(precision=precision, alwayssign=alwayssign)
        p = Phase.from_string(s)
        # We cannot get exact round-tripping, since we treat fractional
        # near 0 as if it is near 0.25.
        assert np.allclose((p - p_in).value, 0, atol=2**-53, rtol=0)

    def test_to_from_string_alwayssign(self):
        ph = Phase([[-10], [20]], [-0.4, 0.4])
        s = ph.to_string(alwayssign=True)
        ph2 = Phase.from_string(s)
        # No worries about round-off, since fractions relatively large.
        assert np.all(ph == ph2)

    @pytest.mark.parametrize('item', (123456789., '12.34.56', '12.34e5.',
                                      '12.34e5e1', '12.34je-1', '10+11j'))
    def test_from_string_invalid(self, item):
        with pytest.raises(ValueError):
            Phase.from_string(item)


class TestFractionalPhase(PhaseSetup):
    """Since FractionalPhase is a subclass of Longitude, only limited tests."""

    def test_keep_precision(self):
        fp = FractionalPhase(self.phase)
        assert np.all(fp == self.phase2)
