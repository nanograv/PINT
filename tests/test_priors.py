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
        # Call prior_pdf using parameter values set in model
        assert self.m.F0.prior_pdf() == 1.0
        assert self.m.F0.prior_pdf(logpdf=True) == 0.0
        
        # Now try overriding the parameter values
        assert self.m.F0.prior_pdf(10.0) == 1.0
        assert self.m.F0.prior_pdf(10.0, logpdf=True) == 0.0
        
    def test_uniform_bounded(self):
        print("test_uniform_bounded")
        self.m.ECC.prior = Prior(UniformBoundedRV(0.0,1.0))
        assert self.m.ECC.prior_pdf() == 1.0
        assert self.m.ECC.prior_pdf(logpdf=True) == 0.0
        assert self.m.ECC.prior_pdf(-0.1) == 0.0
        assert self.m.ECC.prior_pdf(-0.1,logpdf=True) == -np.inf
        assert self.m.ECC.prior_pdf(1.1) == 0.0
        assert self.m.ECC.prior_pdf(1.1,logpdf=True) == -np.inf
        
    def test_gaussian(self):
        print("test_gaussian")
        v = -6.2e-16
        s = 1.0e-18
        tv = self.m.F1.value
        self.m.F1.prior = Prior(norm(loc=v,scale=s))
        print(self.m.F1.prior_pdf(v))
        assert numpy.isclose(self.m.F1.prior_pdf(v), 1.0/(s*np.sqrt(2.0*np.pi)))
    

if __name__ == '__main__':
    unittest.main()
