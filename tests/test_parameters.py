import copy
import os
import pickle
import unittest

import astropy.time as time
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pinttestdata import datadir

import pint.fitter
from pint import pint_units
from pint.models.model_builder import get_model
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    boolParameter,
    floatParameter,
    intParameter,
    maskParameter,
    pairParameter,
    prefixParameter,
    strParameter,
)
from pint.toa import get_TOAs

# FIXME: this should be in the docs!
"""
The behavior we want for numerical Parameter variables (p):

    p.unit returns the astropy Unit

    str(p.unit) is the string version

    p.quantity returns the Quantity.

    p.value returns the numerical value (without units).

    p.value = new_value stores the new value, assuming it is given in
    the current p.unit.

    p.value = new_value*u.NewCompatibleUnit converts new_value to the
    existing p.unit and stores this as the value.  Does not change
    p.unit.

    p.quantity = (anything) acts the same as setting p.value

    p.uncertainty acts analagous to p.quantity.  p.uncertainty_value acts
    like p.value.

    p.unit = u.NewCompatibleUnit changes all internal representations of
    value, uncertaintly and units to the new units.

"""


def test_read_par_line_expected_values():
    test_m = get_model(os.path.join(datadir, "test_par_read.par"))
    assert test_m.F2.frozen
    assert test_m.F3.frozen
    assert_allclose(test_m.F3.value, 0.0)
    assert_allclose(test_m.F3.uncertainty_value, 0.0)
    assert test_m.F4.frozen
    assert_allclose(test_m.F4.value, 0.0)
    assert_allclose(test_m.F4.uncertainty_value, 0.001)
    assert test_m.F5.frozen
    assert_allclose(test_m.F5.value, 0.0)
    assert_allclose(test_m.F5.uncertainty_value, 0.001)
    assert not test_m.F6.frozen
    assert_allclose(test_m.F6.value, 0.0)
    assert_allclose(test_m.F6.uncertainty_value, 0.001)
    assert test_m.F7.frozen
    assert_allclose(test_m.F7.value, 0.0)
    assert_allclose(test_m.F7.uncertainty_value, 3.0)
    assert test_m.F8.frozen
    assert_allclose(test_m.F8.value, 0.0)
    assert_allclose(test_m.F8.uncertainty_value, 10)
    assert test_m.JUMP1.frozen
    assert test_m.JUMP1.key == "MJD"
    assert_allclose(test_m.JUMP1.key_value[0], 52742.0, atol=1e-10)
    assert_allclose(test_m.JUMP1.key_value[1], 52745.0, atol=1e-10)
    assert_allclose(test_m.JUMP1.value, 0.2)

    assert test_m.JUMP2.frozen
    assert_allclose(test_m.JUMP2.value, 0.1)
    assert_allclose(test_m.JUMP2.uncertainty_value, 0.0)
    assert_allclose(test_m.JUMP7.value, 0.1)
    assert_allclose(test_m.JUMP7.uncertainty_value, 10.5)
    assert_allclose(test_m.JUMP6.value, 0.1)
    assert_allclose(test_m.JUMP6.uncertainty_value, 10.0)
    assert test_m.JUMP12.key == "-testflag"
    assert not test_m.JUMP12.frozen
    assert test_m.JUMP12.key_value[0] == "flagvalue"
    assert_allclose(test_m.JUMP12.value, 0.1)
    assert_allclose(test_m.JUMP12.uncertainty_value, 2.0)
    assert_allclose(
        test_m.RAJ.uncertainty,
        476.94611148516092061223 * pint_units["hourangle_second"],
    )

    assert_allclose(
        test_m.DECJ.uncertainty,
        190996312986311097848351.00000000000000000000 * u.arcsecond,
    )


@pytest.mark.xfail(
    reason="For some reason AngleParameters should have different units for their uncertainties"
)
def test_read_par_line_raj_units():
    test_m = get_model(os.path.join(datadir, "test_par_read.par"))
    assert test_m.RAJ.uncertainty.unit == pint_units["hourangle_second"]
    assert test_m.DECJ.uncertainty.unit == u.arcsecond


@pytest.mark.xfail(
    reason="For some reason AngleParameters should have different units for their uncertainties"
)
def test_read_par_line_decj_units():
    test_m = get_model(os.path.join(datadir, "test_par_read.par"))
    assert test_m.DECJ.uncertainty.unit == u.arcsecond


def test_units_consistent():
    m = get_model(os.path.join(datadir, "test_par_read.par"))
    for p in m.params:
        pm = getattr(m, p)
        if not hasattr(pm.quantity, "unit"):
            continue
        assert pm.quantity.unit == pm.units
        assert pm.quantity == pm.value * pm.units
        if pm.uncertainty is None:
            continue
        assert pm.uncertainty.unit == pm.units
        assert pm.uncertainty_value * pm.units == pm.uncertainty


class TestParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.m = get_model("B1855+09_NANOGrav_dfg+12_modified.par")
        cls.mp = get_model("prefixtest.par")

    def test_RAJ(self):
        """Check whether the value and units of RAJ parameter are ok"""
        units = u.hourangle
        value = 18.960109246777776

        self.assertEqual(self.m.RAJ.units, units)
        self.assertEqual(self.m.RAJ.value, value)
        self.assertEqual(self.m.RAJ.quantity, value * units)

    def test_DECJ(self):
        """Check whether the value and units of DECJ parameter are ok"""
        units = u.deg
        value = 9.72146998888889
        self.assertEqual(self.m.DECJ.units, units)
        self.assertEqual(self.m.DECJ.value, value)
        self.assertEqual(self.m.DECJ.quantity, value * units)

    def test_F0(self):
        """Check whether the value and units of F0 parameter are ok"""
        units = u.Hz
        value = np.longdouble("186.49408156698235146")

        self.assertEqual(self.m.F0.units, units)
        self.assertTrue(np.isclose(self.m.F0.value, value, atol=1e-19))
        self.assertEqual(self.m.F0.value, value)

    def test_F0_uncertainty(self):
        uncertainty = 0.00000000000698911818
        units = self.m.F0.units
        # Test stored uncertainty value
        self.assertTrue(
            np.isclose(self.m.F0.uncertainty.to(units).value, uncertainty, atol=1e-20)
        )
        # Test parameter.uncertainty_value returned value
        self.assertTrue(
            np.isclose(self.m.F0.uncertainty_value, uncertainty, atol=1e-20)
        )

    def test_set_new_units(self):
        """Check whether we can set the units to non-standard ones """
        # Standard units
        units = u.Hz
        str_unit = "Hz"
        value = 186.49
        uncertainty_value = 0.00000000000698911818
        # New units
        units_new = u.kHz
        str_unit_new = "kHz"
        value_new = 0.18649
        uncertainty_value_new = 0.00000000000698911818 / 1000.0
        # Set it to 186.49 Hz: the 'standard' value
        self.m.F0.quantity = value * units
        self.assertTrue(np.isclose(self.m.F0.value, value, atol=1e-13))

        # Now change the units
        self.m.F0.units = str_unit_new
        self.assertTrue(np.isclose(self.m.F0.value, value_new, atol=1e-13))

        self.assertTrue(
            np.isclose(self.m.F0.uncertainty_value, uncertainty_value_new, atol=1e-13)
        )
        # Change the units back, and then set them implicitly
        # The value will be associate with the new units
        self.m.F0.units = str_unit
        self.m.F0.quantity = value_new * units_new
        self.m.F0.uncertainty = uncertainty_value_new * units_new
        self.assertTrue(np.isclose(self.m.F0.value, value, atol=1e-13))
        self.assertTrue(
            np.isclose(self.m.F0.uncertainty_value, uncertainty_value, atol=1e-20)
        )
        # Check the ratio, using the old units as a reference
        ratio = self.m.F0.quantity / (value * units)
        ratio_uncertainty = self.m.F0.uncertainty / (uncertainty_value * units)
        self.assertTrue(np.isclose(ratio.decompose(u.si.bases), 1.0, atol=1e-13))

        self.assertTrue(
            np.isclose(ratio_uncertainty.decompose(u.si.bases), 1.0, atol=1e-20)
        )

    def set_units_fail(self):
        """Setting the unit to a non-compatible unit should fail """
        self.m.RAJ.units = u.m

    def test_units(self):
        """Test setting the units """
        self.assertRaises(u.UnitConversionError, self.set_units_fail)

    def set_num_to_unit(self):
        """Try to set the numerical value to a unit """
        self.m.RAJ.value = u.m

    def set_num_to_quantity(self):
        """Try to set the numerical value to a quantity """
        self.m.RAJ.value = 1.0 * u.m

    def test_set_value(self):
        """Try to set the numerical value of a parameter to various things"""
        self.assertRaises(ValueError, self.set_num_to_unit)
        self.assertRaises(ValueError, self.set_num_to_quantity)

    def test_T0(self):
        """Test setting T0 to a test value"""
        self.m.T0.value = 50044.3322
        # I don't understand why this is failing...  something about float128
        # Does not fail for me (both lines)  -- RvH 02/22/2015
        self.assertTrue(np.isclose(self.m.T0.value, 50044.3322))
        self.assertEqual(self.m.T0.value, 50044.3322)

    def set_num_to_none(self):
        """Set T0 to None"""
        self.m.T0.value = None

    def set_num_to_string(self):
        """Set T0 to a string"""
        self.m.T0.value = "this is a string"

    def test_num_to_other(self):
        """Test setting the T0 numerical value to a not-number"""
        self.assertRaises(ValueError, self.set_num_to_none)
        self.assertRaises(ValueError, self.set_num_to_string)

    def set_OM_to_none(self):
        """Set OM to None"""
        self.m.OM.value = None

    def set_OM_to_time(self):
        """Set OM to a time"""
        self.m.OM.value = time.Time(54000, format="mjd")

    def test_OM(self):
        """Test doing stuff to OM"""
        quantity = 10.0 * u.deg
        self.m.OM.quantity = quantity

        self.assertEqual(self.m.OM.quantity, quantity)
        self.assertRaises(ValueError, self.set_OM_to_none)
        self.assertRaises(TypeError, self.set_OM_to_time)

    def test_PBDOT(self):
        # Check that parameter scaling is working as expected
        # Units are not modified, just the value is scaled
        self.m.PBDOT.value = 20
        self.assertEqual(self.m.PBDOT.units, u.day / u.day)
        self.assertEqual(self.m.PBDOT.quantity, 20 * 1e-12 * u.day / u.day)
        self.m.PBDOT.value = 1e-11
        self.assertEqual(self.m.PBDOT.units, u.day / u.day)
        self.assertEqual(self.m.PBDOT.quantity, 1e-11 * u.day / u.day)

    def test_prefix_value_to_num(self):
        """Test setting the prefix parameter """
        value = 51
        units = u.Hz
        self.mp.GLF0_2.value = value

        self.assertEqual(self.mp.GLF0_2.quantity, value * units)

        value = 50
        self.mp.GLF0_2.value = value
        self.assertEqual(self.mp.GLF0_2.quantity, value * units)

    def test_prefix_value_str(self):
        """Test setting the prefix parameter from a string"""
        str_value = "50"
        value = 50
        units = u.Hz

        self.mp.GLF0_2.value = str_value

        self.assertEqual(self.mp.GLF0_2.value, value * units)

    def set_prefix_value_to_unit_fail(self):
        """Set the prefix parameter to an incompatible value"""
        value = 50
        units = u.s

        self.mp.GLF0_2.value = value * units

    def test_prefix_value_fail(self):
        """Test setting the prefix parameter to an incompatible value"""
        self.assertRaises(ValueError, self.set_prefix_value_to_unit_fail)

    def test_prefix_value1(self):
        self.mp.GLF0_2.value = 50
        assert self.mp.GLF0_2.quantity == 50 * u.Hz

    def test_prefix_value_str(self):
        self.mp.GLF0_2.value = "50"
        assert self.mp.GLF0_2.quantity == 50 * u.Hz

    def test_prefix_value_quantity(self):
        self.mp.GLF0_2.value = 50 * u.Hz
        assert self.mp.GLF0_2.quantity == 50 * u.Hz

    def set_prefix_value1(self):
        self.mp.GLF0_2.value = 100 * u.s

    def test_prefix_value1(self):
        self.assertRaises(ValueError, self.set_prefix_value1)

    def test_START_FINISH_in_par(self):
        """
        Check that START/FINISH parameters set up/operate properly when
        from input file.
        """
        m1 = self.m
        t1 = get_TOAs("B1855+09_NANOGrav_dfg+12.tim")

        start_preval = 53358.726464889485214
        finish_preval = 55108.922917417192366
        start_postval = 53358.7274648895  # from Tempo2
        finish_postval = 55108.9219174172  # from Tempo2

        # check parameter initialization
        assert hasattr(m1, "START")
        assert isinstance(m1.START, MJDParameter)
        assert hasattr(m1, "FINISH")
        assert isinstance(m1.FINISH, MJDParameter)

        self.assertEqual(m1.START.value, start_preval)
        self.assertEqual(m1.FINISH.value, finish_preval)
        self.assertEqual(m1.START.frozen, True)
        self.assertEqual(m1.FINISH.frozen, True)

        # fit toas and compare with expected/Tempo2 (for WLS) values
        fitters = [
            pint.fitter.WLSFitter(toas=t1, model=m1),
            pint.fitter.GLSFitter(toas=t1, model=m1),
        ]
        for fitter in fitters:
            fitter.fit_toas()
            self.assertEqual(m1.START.frozen, True)
            self.assertEqual(m1.FINISH.frozen, True)
            if fitter.method == "weighted_least_square":
                self.assertAlmostEqual(
                    fitter.model.START.value, start_postval, places=9
                )
                self.assertAlmostEqual(
                    fitter.model.FINISH.value, finish_postval, places=9
                )
            self.assertAlmostEqual(
                fitter.model.START.value, fitter.toas.first_MJD.value, places=9
            )
            self.assertAlmostEqual(
                fitter.model.FINISH.value, fitter.toas.last_MJD.value, places=9
            )

    def test_START_FINISH_not_in_par(self):
        """
        Check that START/FINISH parameters are added and set up when not
        in input file.
        """
        # check initialization after fitting for .par file without START/FINISH
        m = get_model("NGC6440E.par")
        t = get_TOAs("NGC6440E.tim")

        start_postval = 53478.2858714192  # from Tempo2
        finish_postval = 54187.5873241699  # from Tempo2

        self.assertTrue(hasattr(m, "START"))
        self.assertTrue(hasattr(m, "FINISH"))

        # fit toas and compare with expected/Tempo2 (for WLS) values
        fitters = [
            pint.fitter.WLSFitter(toas=t, model=m),
            pint.fitter.GLSFitter(toas=t, model=m),
        ]
        for fitter in fitters:
            fitter.fit_toas()
            self.assertTrue(hasattr(fitter.model, "START"))
            self.assertTrue(hasattr(fitter.model, "FINISH"))
            self.assertEqual(fitter.model.START.frozen, True)
            self.assertEqual(fitter.model.FINISH.frozen, True)
            if fitter.method == "weighted_least_square":
                self.assertAlmostEqual(
                    fitter.model.START.value, start_postval, places=9
                )
                self.assertAlmostEqual(
                    fitter.model.FINISH.value, finish_postval, places=9
                )
            self.assertAlmostEqual(
                fitter.model.START.value, fitter.toas.first_MJD.value, places=9
            )
            self.assertAlmostEqual(
                fitter.model.FINISH.value, fitter.toas.last_MJD.value, places=9
            )


@pytest.mark.parametrize(
    "type_,str_value,value",
    [
        (floatParameter, "TEST 1.234", 1.234),
        (floatParameter, "TEST 1.234e8", 1.234e8),
        (floatParameter, "TEST 1.234d8", 1.234e8),
        (boolParameter, "TEST Y", True),
        (boolParameter, "TEST N", False),
        (boolParameter, "TEST 1", True),
        (boolParameter, "TEST 0", False),
        (boolParameter, "TEST 1.0", True),
        (boolParameter, "TEST 0.0", False),
        (boolParameter, "TEST 3.14", True),
        (boolParameter, "TEST nan", True),
        (boolParameter, "TEST True", True),
        (boolParameter, "TEST False", False),
        (boolParameter, "TEST y", True),
        (boolParameter, "TEST n", False),
        (intParameter, "TEST 17", 17),
        (intParameter, "TEST 17.0", 17),
        (intParameter, "TEST -17", -17),
        (
            intParameter,
            "TEST -17_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000",
            -17_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000,
        ),
        (strParameter, "TEST FISH", "FISH"),
        (strParameter, "TEST Fish", "Fish"),
    ],
)
def test_parameter_parsing(type_, str_value, value):
    p = type_(name="TEST")
    assert p.from_parfile_line(str_value)
    assert p.value == value


@pytest.mark.parametrize(
    "type_,par_line",
    [
        (floatParameter, "TEST fish"),
        (floatParameter, "TEST 1.234h500"),
        (floatParameter, "TEST 0x1234"),
        (intParameter, "TEST 1.5"),
        (intParameter, "TEST fish"),
        (intParameter, "TEST 5fish"),
        (boolParameter, "TEST FRUE"),
    ],
)
def test_parameter_parsing_fail(type_, par_line):
    p = type_(name="TEST")
    with pytest.raises(ValueError):
        p.from_parfile_line(par_line)


@pytest.mark.parametrize(
    "type_,set_value,value",
    [
        (floatParameter, 1.234, 1.234),
        (floatParameter, 1, 1.0),
        (floatParameter, "1.234", 1.234),
        (floatParameter, "1.234q8", ValueError),
        (floatParameter, [1, 2], TypeError),
        (intParameter, 2, 2),
        (intParameter, "2", 2),
        (intParameter, 2.0, 2),
        (intParameter, 1.5, ValueError),
    ],
)
def test_parameter_setting(type_, set_value, value):
    p = type_(name="TEST")
    if isinstance(value, type(Exception)) and issubclass(value, Exception):
        with pytest.raises(value):
            p.value = set_value
    else:
        p.value = set_value
        assert p.value == value


@pytest.mark.parametrize(
    "p",
    [boolParameter(name="FISH"), intParameter(name="FISH"), strParameter(name="FISH")],
)
def test_set_uncertainty_bogus_raises(p):
    with pytest.raises(NotImplementedError):
        p.uncertainty = 7.0


def test_compare_key_value():
    par1 = maskParameter(name="Test1", key='-k1', key_value='kv1')
    par2 = maskParameter(name="Test2", key='-k1', key_value='kv1')
    par3 = maskParameter(name="Test3", key='-k2', key_value='kv1')
    par4 = maskParameter(name="Test4", key='-k1', key_value='kv2')
    assert par1.compare_key_value(par2)
    assert par2.compare_key_value(par1)
    assert (not par1.compare_key_value(par3))
    assert (not par1.compare_key_value(par4))
    assert (not par2.compare_key_value(par3))
    assert (not par2.compare_key_value(par4))

def test_compare_key_value_list():
    par1 = maskParameter(name="Test1", key='-k1', key_value=['k', 'q'])
    par2 = maskParameter(name="Test2", key='-k1', key_value=['q', 'k'])
    par3 = maskParameter(name="Test3", key='-k1', key_value=['k', 'q'])
    assert par1.compare_key_value(par2)
    assert par2.compare_key_value(par1)
    assert par3.compare_key_value(par1)
    par4 = maskParameter(name="Test4", key='-k1', key_value=[2000, 3000])
    par5 = maskParameter(name="Test5", key='-k1', key_value=[3000, 2000])
    assert par4.compare_key_value(par5)
    assert par5.compare_key_value(par4)
    par6 = maskParameter(name="Test6", key='-k1', key_value=['430', 'guppi', 'puppi'])
    par7 = maskParameter(name="Test7", key='-k1', key_value=['puppi', 'guppi', '430'])
    assert par6.compare_key_value(par7)
    assert par7.compare_key_value(par6)


@pytest.mark.parametrize(
    "p",
    [
        boolParameter(name="FISH"),
        intParameter(name="FISH"),
        strParameter(name="FISH"),
        pytest.param(maskParameter(name="JUMP"),),
        pytest.param(prefixParameter(name="F0"),),
        pairParameter(name="WEAVE"),
        AngleParameter(name="BEND"),
    ],
)
def test_parameter_can_be_pickled(p):
    pickle.dumps(p)
