"""Various tests to assess the performance of the PINT position.
"""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
import test_derivative_utils as tdu
import logging

from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestPulsarPosition(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # This uses ELONG and ELAT
        self.m1 = mb.get_model('B1855+09_NANOGrav_9yv1.gls.par')
        # This uses RA and DEC
        self.m2 = mb.get_model('B1855+09_NANOGrav_dfg+12_TAI_FB90.par')
        self.t = 5000 * np.random.randn(100) + 49453.0

    def test_ssb_2_psr(self):
        PMELONG_v = self.m1.PMELONG.value
        PMELAT_v = self.m1.PMELAT.value
        PMRA = self.m2.PMRA.value
        PMDEC = self.m2.PMDEC.value
        # Switch off PM
        self.m1.PMELONG.value = 0.0
        self.m1.PMELAT.value = 0.0
        self.m2.PMRA.value = 0.0
        self.m2.PMDEC.value = 0.0

        p1 = self.m1.ssb_to_psb_xyz(epoch = self.t)
        p2 = self.m2.ssb_to_psb_xyz(epoch = self.t)

        self.assertTrue(np.max(np.abs(p1-p2))<1e-6)

        # Switch on PM
        self.m1.PMELONG.value = PMELONG_v
        self.m1.PMELAT.value = PMELAT_v
        self.m2.PMRA.value = PMRA
        self.m2.PMDEC.value = PMDEC

        p1 = self.m1.ssb_to_psb_xyz(epoch = self.t)
        p2 = self.m2.ssb_to_psb_xyz(epoch = self.t)

        self.assertTrue(np.max(np.abs(p1-p2))<1e-7)

    def test_parse_line(self):
        self.m1.ELONG.from_parfile_line('LAMBDA   286.8634893301156  1  0.0000000165859')
        self.m1.ELAT.from_parfile_line('BETA      32.3214877555037  1   0.0000000273526')
        self.m1.PMELONG.from_parfile_line('PMLAMBDA  -3.2701  1 0.0141')
        self.m1.PMELAT.from_parfile_line('PMBETA  -5.0982  1  0.0291')
        ELONG_v = self.m1.ELONG.value
        ELAT_v = self.m1.ELAT.value
        PMELONG_v = self.m1.PMELONG.value
        PMELAT_v = self.m1.PMELAT.value

        self.m1.ELONG.from_parfile_line('ELONG   286.8634893301156  1  0.0000000165859')
        self.m1.ELAT.from_parfile_line('ELAT      32.3214877555037  1   0.0000000273526')
        self.m1.PMELONG.from_parfile_line('PMELONG  -3.2701  1 0.0141')
        self.m1.PMELAT.from_parfile_line('PMELAT  -5.0982  1  0.0291')

        self.assertTrue(np.isclose(self.m1.ELONG.value, ELONG_v))
        self.assertTrue(np.isclose(self.m1.ELAT.value, ELAT_v))
        self.assertTrue(np.isclose(self.m1.PMELONG.value, PMELONG_v))
        self.assertTrue(np.isclose(self.m1.PMELAT.value, PMELAT_v))
