#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import unittest
import pint.models
from pint.models.priors import *
from scipy.stats import norm
import os

testdir=os.path.join(os.getenv('PINT'),'tests');
datadir = os.path.join(testdir,'datafile')
os.chdir(datadir)

class TestPriors(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = pint.models.get_model('B1855+09_NANOGrav_dfg+12_modified.par')


    def test_uniform(self):
        print("test_uniform")
        assert self.m.F0.prior_pdf() == 1.0
        assert self.m.F0.prior_pdf(logpdf=True) == 0.0

#    def test_uniform_bounded(self):
#        print("test_uniform_bounded")    

if __name__ == '__main__':
    unittest.main()
