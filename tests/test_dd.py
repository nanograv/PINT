"""Various tests to assess the performance of the DD model."""
import pint.models.model_builder as mb
import pint.toa as toa
import libstempo as lt
import matplotlib.pyplot as plt
import tempo2_utils
import astropy.units as u
from pint.residuals import resids
from pint.models.pint_dd_model import DDwrapper
import numpy as np
import os, unittest

from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestDD(unittest.TestCase):
    """Compare delays from the dd model with libstempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileB1855 = 'B1855+09_NANOGrav_dfg+12_modified.par'
        self.timB1855 = 'B1855+09_NANOGrav_dfg+12.tim'
        self.toasB1855 = toa.get_TOAs(self.timB1855, ephem="DE405", planets=False)
        self.modelB1855 = mb.get_model(self.parfileB1855)
        self.psrB1855 = lt.tempopulsar(self.parfileB1855, self.timB1855)
        self.ltres = self.psrB1855.residuals()
    def test_J1855_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelB1855.binary_delay(self.toasB1855.table)

        lt_binary_delay = self.psrB1855.binarydelay()
        assert np.all(np.abs(pint_binary_delay.value + lt_binary_delay) < 1e-11), 'DD J1955 TEST FAILED'
    # TODO: PINT can still incresase the precision by adding more components
    def test_B1855(self):
        pint_resids_s = resids(self.toasB1855, self.modelB1855).time_resids.to(u.s)
        toas = self.psrB1855.toas()
        assert np.all(np.abs(pint_resids_s.value - self.ltres) < 1e-7), 'DD B1855 TEST FAILED'

if __name__ == '__main__':
    pass
