"""Various tests to assess the performance of the B1855+09."""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest

from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestB1855(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileB1855 = 'B1855+09_NANOGrav_dfg+12_TAI_FB90.par'
        self.timB1855 = 'B1855+09_NANOGrav_dfg+12.tim'
        self.toasB1855 = toa.get_TOAs(self.timB1855, ephem="DE405", planets=False)
        self.modelB1855 = mb.get_model(self.parfileB1855)
        # tempo result
        self.ltres, self.ltbindelay = np.genfromtxt(self.parfileB1855 + \
                                     '.tempo2_test',skip_header=1, unpack=True)
        print self.ltres
    def test_B1855_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelB1855.binarymodel_delay(self.toasB1855.table)
        assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-8), 'B1855 binary delay test failed.'

    def test_B1855(self):
        pint_resids_us = resids(self.toasB1855, self.modelB1855).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level. 
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 3e-8), 'B1855 residuals test failed.'


if __name__ == '__main__':
    pass
