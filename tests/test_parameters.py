from pint.models import parameter as p
from pint.models import model_builder as mb
from pint.utils import str2longdouble
from astropy.coordinates.angles import Angle
import astropy.time as time
import astropy.units as u
import numpy, os, unittest

testdir=os.path.join(os.getenv('PINT'),'tests');
datadir = os.path.join(testdir,'datafile')
os.chdir(datadir)

class TestParameters(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = mb.get_model('B1855+09_NANOGrav_dfg+12_modified.par')
        self.mp = mb.get_model('prefixtest.par')
    def test_RAJ(self):
        """Check whether the value and units of RAJ parameter are ok"""
        num_unit = u.hourangle
        num_value = 18.960109246777776

        self.assertEqual(self.m.RAJ.num_unit, num_unit)
        self.assertEqual(self.m.RAJ.num_value, num_value)
        self.assertEqual(self.m.RAJ.value, num_value * num_unit)

    def test_DECJ(self):
        """Check whether the value and units of DECJ parameter are ok"""
        num_unit = u.deg
        num_value = 9.72146998888889
        self.assertEqual(self.m.DECJ.num_unit, num_unit)
        self.assertEqual(self.m.DECJ.num_value, num_value)
        self.assertEqual(self.m.DECJ.value, num_value*num_unit)

    def test_F0(self):
        """Check whether the value and units of F0 parameter are ok"""
        num_unit = u.Hz
        num_value = str2longdouble('186.49408156698235146')
        num_value = 186.49408156698235

        self.assertEqual(self.m.F0.num_unit, num_unit)
        self.assertTrue(
                numpy.isclose(self.m.F0.num_value, num_value, atol=1e-13))
        self.assertEqual(self.m.F0.num_value, num_value)

    def test_F0_uncertainty(self):
        uncertainty = 0.00000000000698911818
        num_unit = self.m.F0.num_unit
        self.assertEqual(self.m.F0.uncertainty, uncertainty * num_unit)

    def test_set_new_units(self):
        """Check whether we can set the units to non-standard ones """
        # Standard units
        num_unit = u.Hz
        str_unit = 'Hz'
        num_value = 186.49
        num_uncertainty = 0.00000000000698911818
        # New units
        num_unit_new = u.kHz
        str_unit_new = 'kHz'
        num_value_new = 0.18649
        num_uncertainty_new = 0.00000000000698911818/1000.0
        # Set it to 186.49 Hz: the 'standard' value
        self.m.F0.value = num_value * num_unit
        self.assertTrue(
                numpy.isclose(self.m.F0.num_value, num_value, atol=1e-13))

        # Now change the units
        self.m.F0.units = str_unit_new
        self.assertTrue(
                numpy.isclose(self.m.F0.num_value, num_value_new, atol=1e-13))

        self.assertTrue(
                numpy.isclose(self.m.F0.num_uncertainty, num_uncertainty_new,
                              atol=1e-13))
        # Change the units back, and then set them implicitly
        self.m.F0.units = str_unit
        self.m.F0.value = num_value_new * num_unit_new
        self.m.F0.uncertainty = num_uncertainty_new * num_unit_new
        self.assertTrue(
                numpy.isclose(self.m.F0.num_value, num_value_new, atol=1e-13))
        self.assertTrue(
                numpy.isclose(self.m.F0.num_uncertainty, num_uncertainty_new,
                              atol=1e-13))
        # Check the ratio, using the old units as a reference
        ratio = self.m.F0.value / (num_value * num_unit)
        ratio_uncertainty = self.m.F0.uncertainty / (num_uncertainty * num_unit)
        self.assertTrue(
                numpy.isclose(ratio.decompose(u.si.bases), 1.0, atol=1e-13))

        self.assertTrue(
                numpy.isclose(ratio_uncertainty.decompose(u.si.bases), 1.0,
                              atol=1e-13))

    def set_num_unit_fail(self):
        """Setting the unit to a non-compatible unit should fail """
        self.m.RAJ.num_unit = u.m

    def test_num_unit(self):
        """Test setting the units """
        self.assertRaises(AttributeError, self.set_num_unit_fail)

    def set_num_to_unit(self):
        """Try to set the numerical value to a unit """
        self.m.RAJ.num_value = u.m

    def set_num_to_quantity(self):
        """Try to set the numerical value to a quantity """
        self.m.RAJ.num_value = 1.0*u.m

    def test_set_num_value(self):
        """Try to set the numerical value of a parameter to various things"""
        self.assertRaises(ValueError, self.set_num_to_unit)
        self.assertRaises(ValueError, self.set_num_to_quantity)

    def test_T0(self):
        """Test setting T0 to a test value"""
        self.m.T0.num_value = 50044.3322
        # I don't understand why this is failing...  something about float128
        # Does not fail for me (both lines)  -- RvH 02/22/2015
        self.assertTrue(numpy.isclose(self.m.T0.num_value, 50044.3322))
        self.assertEqual(self.m.T0.num_value, 50044.3322)

    def set_num_to_none(self):
        """Set T0 to None"""
        self.m.T0.num_value = None

    def set_num_to_string(self):
        """Set T0 to a string"""
        self.m.T0.num_value = 'this is a string'

    def test_num_to_other(self):
        """Test setting the T0 numerical value to a not-number"""
        self.assertRaises(ValueError, self.set_num_to_none)
        self.assertRaises(ValueError, self.set_num_to_string)

    def set_OM_to_none(self):
        """Set OM to None"""
        self.m.OM.value = None

    def set_OM_to_time(self):
        """Set OM to a time"""
        self.m.OM.value = time.Time(54000,format = 'mjd')

    def test_OM(self):
        """Test doing stuff to OM"""
        value = 10.0 * u.deg
        self.m.OM.value = value

        self.assertEqual(self.m.OM.value, value)
        self.assertRaises(ValueError, self.set_OM_to_none)
        self.assertRaises(ValueError, self.set_OM_to_time)

    def test_prefix_value_to_num(self):
        """Test setting the prefix parameter """
        num_value = 51
        num_unit = u.Hz
        self.mp.GLF0_2.value = num_value

        self.assertEqual(self.mp.GLF0_2.value, num_value * num_unit)

        num_value = 50
        self.mp.GLF0_2.num_value = num_value
        self.assertEqual(self.mp.GLF0_2.value, num_value * num_unit)

    def test_prefix_value_str(self):
        """Test setting the prefix parameter from a string"""
        str_value = '50'
        num_value = 50
        num_unit = u.Hz

        self.mp.GLF0_2.value = str_value

        self.assertEqual(self.mp.GLF0_2.value, num_value * num_unit)

    def set_prefix_value_to_unit_fail(self):
        """Set the prefix parameter to an incompatible value"""
        num_value = 50
        num_unit = u.s

        self.mp.GLF0_2.value = num_value * num_unit

    def test_prefix_value_fail(self):
        """Test setting the prefix parameter to an incompatible value"""
        self.assertRaises(ValueError, self.set_prefix_value_to_unit_fail)

    def test_prefix_value1(self):
        self.mp.GLF0_2.value = 50
        assert self.mp.GLF0_2.value == 50 * u.Hz

    def test_prefix_value_str(self):
        self.mp.GLF0_2.value = '50'
        assert self.mp.GLF0_2.value == 50 * u.Hz

    def test_prefix_value_quantity(self):
        self.mp.GLF0_2.value = 50 * u.Hz
        assert self.mp.GLF0_2.value == 50 * u.Hz

    def set_prefix_value1(self):
        self.mp.GLF0_2.value = 100 * u.s
    def test_prefix_value1(self):
        self.assertRaises(ValueError, self.set_prefix_value1)

if __name__ == '__main__':
    pass
