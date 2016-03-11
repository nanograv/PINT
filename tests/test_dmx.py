from pint.models import model_builder as mb
import pint.toa as toa
import matplotlib.pyplot as plt
from pint import residuals
import astropy.units as u
import libstempo as lt
import os
import unittest
import numpy as np

from pinttestdata import testdir, datadir

class TestDMX(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf = os.path.join(datadir, 'B1855+09_NANOGrav_dfg+12_DMX.par')
        self.timf = os.path.join(datadir, 'B1855+09_NANOGrav_dfg+12.tim')
        self.DMXm = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf, ephem='DE405')

    def test_DMX(self):
        print "Testing DMX module."
        rs = residuals.resids(self.toas, self.DMXm).time_resids.to(u.s).value
        psr = lt.tempopulsar(self.parf, self.timf)
        resDiff = rs-psr.residuals()
        assert np.all(np.abs(resDiff) < 5e-5),\
            "PINT and tempo Residual difference is too big."
if __name__ == '__main__':
    pass
