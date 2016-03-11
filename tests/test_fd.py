"""Various tests to assess the performance of the FD model."""
import pint.models.model_builder as mb
import pint.toa as toa
import libstempo as lt
import matplotlib.pyplot as plt
import tempo2_utils
import astropy.units as u
import pint.residuals
import numpy as np
import os, unittest
datapath = os.path.join(os.environ['PINT'], 'tests', 'datafile')

class TestFD(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf = os.path.join(datapath, 'test_FD.par')
        self.timf = os.path.join(datapath, 'test_FD.simulate.pint_corrected')
        self.FDm = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf)

    def test_DMX(self):
        print "Testing FD module."
        rs = pint.residuals.resids(self.toas, self.FDm).time_resids.to(u.s).value
        psr = lt.tempopulsar(self.parf, self.timf)
        resDiff = rs-psr.residuals()
        #TODO : this percision should be increased. 
        assert np.all(resDiff< 5e-6) , \
            "PINT and tempo Residual difference is too big. "

if __name__ == '__main__':
    pass
