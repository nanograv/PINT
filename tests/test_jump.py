"""Tests for jump model component """
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
from pinttestdata import testdir, datadir

os.chdir(datadir)


class TestJUMP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf = 'B1855+09_NANOGrav_dfg+12_TAI.par'
        self.timf = 'B1855+09_NANOGrav_dfg+12.tim'
        self.JUMPm = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf, ephem="DE405", planets=False)
        # libstempo calculation
        self.ltres, self.ltbindelay = np.genfromtxt(self.parf + '.tempo_test', unpack=True)
    def test_jump(self):
        presids_s = resids(self.toas, self.JUMPm).time_resids.to(u.s)
        assert np.all(np.abs(presids_s.value - self.ltres) < 1e-7), "JUMP test failed."
