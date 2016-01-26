from pint.models import parameter as p
from pint.models import model_builder as mb
from astropy.coordinates.angles import Angle
import astropy.units as u
import os, unittest

testdir=os.path.join(os.getenv('PINT'),'tests');
os.chdir(testdir)

class TestParameters(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = mb.get_model('B1855+09_NANOGrav_dfg+12_modified.par')
    def test_RAJ(self):
        print "RAJ"
        assert self.m.RAJ.base_unit == u.hourangle
        assert self.m.RAJ.base_value == 18.960109246777776
    def test_DECJ(self):
        assert self.m.DECJ.base_unit == u.deg
        assert self.m.DECJ.base_value == 9.72146998888889
    def set_base_unit(self):
        self.m.RAJ.base_unit = u.m
    def test_base_unit(self):
        self.assertRaises(AttributeError, self.set_base_unit)
    def set_base_value(self):
        self.m.RAJ.base_value = u.m
    def test_base_value(self):
        self.assertRaises(ValueError, self.set_base_value)

if __name__ == '__main__':
    pass
