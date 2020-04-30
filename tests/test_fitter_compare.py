#! /usr/bin/env python
import os
import unittest

from pint.models.model_builder import get_model
from pint import toa
from pint.fitter import WLSFitter, GLSFitter
from pinttestdata import datadir


class TestFitterCompare(unittest.TestCase):
    """Compare results from WLS and GLS fitters."""

    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.par = "NGC6440E.par"
        cls.tim = "NGC6440E.tim"
        cls.m = get_model(cls.par)
        cls.t = toa.get_TOAs(cls.tim, ephem="DE421")
        cls.wls = WLSFitter(cls.t, cls.m)
        cls.gls = GLSFitter(cls.t, cls.m)
        cls.gls_full = GLSFitter(cls.t, cls.m)

    def test_compare(self):
        self.wls_chi2 = self.wls.fit_toas()
        self.gls_chi2 = self.gls.fit_toas()
        self.gls_full_chi2 = self.gls.fit_toas(full_cov=True)
        assert abs((self.wls_chi2 - self.gls_chi2) < 0.01)
        assert abs((self.wls_chi2 - self.gls_full_chi2) < 0.01)
        assert abs((self.gls_chi2 - self.gls_full_chi2) < 0.01)
