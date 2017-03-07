"""Various tests to assess the performance of the FD model."""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
import pint.residuals
import numpy as np
import os, unittest
from pinttestdata import testdir, datadir
os.chdir(datadir)

class TestFD(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf = 'test_FD.par'
        self.timf = 'test_FD.simulate.pint_corrected'
        self.FDm = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf, include_bipm=False)
        # libstempo result
        self.ltres, self.ltbindelay = np.genfromtxt(self.parf + '.tempo_test', unpack=True)
    def test_FD(self):
        print("Testing FD module.")
        rs = pint.residuals.resids(self.toas, self.FDm, False).time_resids.to(u.s).value
        resDiff = rs - self.ltres
        #NOTE : This prescision is a lower then 1e-7 seconds level, due to some
        # early parks clock corrections are treated differently.
        # TEMPO2: Clock correction = clock0 + clock1 (in the format of general2)
        # PINT : Clock correction = toas.table['flags']['clkcorr']
        # Those two clock correction difference are causing the trouble.
        assert np.all(resDiff< 5e-6) , \
            "PINT and tempo Residual difference is too big. "

if __name__ == '__main__':
    pass
