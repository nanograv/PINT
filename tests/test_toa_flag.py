"""Various tests to assess the performance of TOA get_flag_value."""


import pint.toa as toa
import astropy.units as u
import numpy as np
import os, unittest
from pinttestdata import testdir, datadir

os.chdir(datadir)

class TestToaFlag(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.tim = 'B1855+09_NANOGrav_dfg+12.tim'
        self.toas = toa.get_TOAs(self.tim, ephem="DE405",
                                      planets=False, include_bipm=False)

    def test_flag_value_float(self):
        flag_value = self.toas.get_flag_value('to')
        assert len(flag_value) == self.toas.ntoas
        for v in set(flag_value):
            assert v in {-7.89e-07, -8.39e-07, None}
        count1 = {}
        for flag_dict in self.toas.get_flags():
            fv1 = flag_dict.get('to')
            if fv1 in list(count1.keys()):
                count1[fv1] += 1
            else:
                count1[fv1] = 1

        count2 = {}
        for fv2 in flag_value:
            if fv2 in list(count2.keys()):
                count2[fv2] += 1
            else:
                count2[fv2] = 1
        assert count1 == count2


    def test_flag_value_fill(self):
        flag_value = self.toas.get_flag_value('to', fill_value=-9999)
        for v in set(flag_value):
            assert v in {-7.89e-07, -8.39e-07, -9999}

    def test_flag_value_str(self):
        flag_value = self.toas.get_flag_value('be')
        assert len(flag_value) == self.toas.ntoas
        for v in set(flag_value):
            assert v in {'ASP', 'PUPPI'}
