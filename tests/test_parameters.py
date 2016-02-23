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

        self.assertEqual(self.m.F0.num_unit, num_unit)
        self.assertEqual(self.m.F0.num_value, num_value)
        self.assertEqual(self.m.F0.value, num_value*num_unit)

    def set_num_unit_fail(self):
        """Setting the unit to a non-compatible unit should fail """
        self.m.RAJ.num_unit = u.m

    def set_num_unit_pass_identical(self):
        """Setting the unit to the same unit as the base unit should pass """
        #self.m.RAJ.num_unit = u.hourangle
        pass

    def set_num_unit_pass_compatible(self):
        """Setting the unit to a compatible unit should pass """
        #self.m.RAJ.num_unit = u.deg
        pass

    def test_num_unit(self):
        """Test setting the units """
        self.assertRaises(AttributeError, self.set_num_unit_fail)
        self.set_num_unit_pass_identical()
        self.set_num_unit_pass_compatible()

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

        self.assertEqual(self.OM.value, value)
        self.assertRaises(ValueError, self.set_OM_to_none)
        self.assertRaises(ValueError, self.set_OM_to_time)

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
