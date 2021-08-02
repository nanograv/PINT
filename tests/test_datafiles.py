"""Test installation of PINT data files"""
import os
import unittest
import pint


class TestDatafiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # get a list of par and tim files to make sure we can find
        files = """B1855+09_NANOGrav_9yv1.gls.par
        B1855+09_NANOGrav_9yv1.tim
        B1855+09_NANOGrav_dfg+12.tim
        B1855+09_NANOGrav_dfg+12_TAI.par
        J0613-sim.par
        J0613-sim.tim
        J1614-2230_NANOGrav_12yv3.wb.gls.par
        J1614-2230_NANOGrav_12yv3.wb.tim
        NGC6440E.par
        NGC6440E.tim
        PSRJ0030+0451_psrcat.par
        waves.par
        waves_withpn.tim"""

        cls.files = [f.strip() for f in files.split("\n")]

    def test_datafiles(self):
        for filename in self.files:
            assert os.path.exists(pint.datafile(filename))
            
