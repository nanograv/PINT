"""Various tests to assess the performance of the J0623-0200."""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest

from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestJ0613(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileJ0613 = 'J0613-0200_NANOGrav_dfg+12_TAI_FB90.par'
        self.timJ0613 = 'J0613-0200_NANOGrav_dfg+12.tim'
        self.toasJ0613 = toa.get_TOAs(self.timJ0613, ephem="DE405", planets=False)
        self.modelJ0613 = mb.get_model(self.parfileJ0613)
        # tempo result
        self.ltres, self.ltbindelay = np.genfromtxt(self.parfileJ0613 + \
                                     '.tempo2_test',skip_header=1, unpack=True)
        print self.ltres
    def test_J0613_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelJ0613.binarymodel_delay(self.toasJ0613.table)
        assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-8), 'J0613 binary delay test failed.'

    def test_J0613(self):
        pint_resids_us = resids(self.toasJ0613, self.modelJ0613).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 3e-8), 'J0613 residuals test failed.'


if __name__ == '__main__':
    pass
