"""Various tests for the maskParameter"""

import os
import unittest

import astropy.time as time
import astropy.units as u
import numpy as np

from pint.models.model_builder import get_model
from pint.models.parameter import maskParameter
from pint.toa import get_TOAs
from pinttestdata import datadir

import copy


class TestParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        toas = get_TOAs("B1855+09_NANOGrav_dfg+12.tim")
        
    def test_mjd_mask(self):
        mp = maskParameter('test1', key='mjd', key_value=[54000, 54100])
        assert mp.key == 'mjd'
        assert mp.key_value == [54000, 54100]
        assert mp.value == None

        mp_str_keyval = maskParameter('test2', key='mjd',
                                      key_value=['54000', '54100'])
        assert mp.key_value == [54000, 54100]

    def test_freq_mask(self):
        pass

    def test_tel_mask(self):
        pass

    def test_name_mask(self):
        pass

    def test_flag_mask(self):
        pass

    def test_key_values(self):
        pass

    def test_read_from_par(self):
        pass
