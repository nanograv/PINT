"""Various tests to assess the performance of the DD model."""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestDD(unittest.TestCase):
    """Compare delays from the dd model with libstempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileB1855 = 'B1855+09_NANOGrav_dfg+12_modified_DD.par'
        self.timB1855 = 'B1855+09_NANOGrav_dfg+12.tim'
        self.toasB1855 = toa.get_TOAs(self.timB1855, ephem="DE405", planets=False)
        self.modelB1855 = mb.get_model(self.parfileB1855)
        # libstempo result
        self.ltres, self.ltbindelay = np.genfromtxt(self.parfileB1855 + '.tempo_test', unpack=True)
    def test_J1855_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelB1855.binarymodel_delay(self.toasB1855.table)
        assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-11), 'DD B1855 TEST FAILED'
    # TODO: PINT can still incresase the precision by adding more components
    def test_B1855(self):
        pint_resids_us = resids(self.toasB1855, self.modelB1855).time_resids.to(u.s)
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 1e-7), 'DD B1855 TEST FAILED'


if __name__ == '__main__':
    pass
