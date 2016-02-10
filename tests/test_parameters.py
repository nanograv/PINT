from pint.models import parameter as p
from pint.models import model_builder as mb
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
        print "RAJ"
        assert self.m.RAJ.num_unit == u.hourangle
        assert self.m.RAJ.num_value == 18.960109246777776

    def test_DECJ(self):
        assert self.m.DECJ.num_unit == u.deg
        assert self.m.DECJ.num_value == 9.72146998888889

    def set_num_unit(self):
        self.m.RAJ.num_unit = u.m
    def test_num_unit(self):
        self.assertRaises(AttributeError, self.set_num_unit)

    def set_num_value1(self):
        self.m.RAJ.num_value = u.m
    def test_num_value1(self):
        self.assertRaises(ValueError, self.set_num_value1)

    def set_num_value2(self):
        self.m.RAJ.num_value = 1*u.m
    def test_num_value2(self):
        self.assertRaises(ValueError, self.set_num_value2)

    def test_num_value3(self):
        self.m.T0.num_value = 50044.3322
        # I don't understand why this is failing...  something about float128
        assert self.m.T0.num_value == 50044.3322

    def set_num_value4(self):
        self.m.T0.num_value = None
    def test_num_value4(self):
        self.assertRaises(ValueError, self.set_num_value4)

    def set_num_value5(self):
        self.m.T0.num_value = 'this is a string'
    def test_num_value5(self):
        self.assertRaises(ValueError, self.set_num_value5)

    def test_value1(self):
        self.m.OM.value = 10
        assert self.m.OM.value == 10 * u.deg

    def set_value2(self):
        self.m.OM.value = None
    def test_value2(self):
        self.assertRaises(ValueError, self.set_value2)

    def set_value3(self):
        self.m.OM.value = time.Time(54000,format = 'mjd')
    def test_value3(self):
        self.assertRaises(ValueError, self.set_value3)

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
