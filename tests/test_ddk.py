"""Various tests to assess the performance of the DD model."""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestDDK(unittest.TestCase):
    """Compare delays from the dd model with libstempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfileJ1713 = 'J1713+0747_NANOGrav_11yv0.gls.par'
        self.timJ1713 = 'J1713+0747_NANOGrav_11yv0_short.tim'
        self.toasJ1713 = toa.get_TOAs(self.timJ1713, ephem="DE421", planets=False)
        self.toasJ1713.table.sort('index')
        self.modelJ1713 = mb.get_model(self.parfileJ1713)
        # libstempo result
        self.ltres, self.ltbindelay = np.genfromtxt(self.parfileJ1713 + '.tempo_test', unpack=True)
    def test_J1713_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelJ1713.binarymodel_delay(self.toasJ1713.table, None)
        assert np.all(np.abs(pint_binary_delay.value + self.ltbindelay) < 8e-11), 'DDK J1713 TEST FAILED'
    def test_J1713(self):
        pint_resids_us = resids(self.toasJ1713, self.modelJ1713, False).time_resids.to(u.s)
        assert np.all(np.abs(pint_resids_us.value - self.ltres) < 1e-8), 'DDK J1713 TEST FAILED'


if __name__ == '__main__':
    pass
