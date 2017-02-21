#! /usr/bin/env python
import time, sys, os, unittest
import pint.models.model_builder as mb
from pint.phase import Phase
from pint import toa
from pint.fitter import WlsFitter
import matplotlib.pyplot as plt
import numpy

from pinttestdata import testdir, datadir

os.chdir(datadir)
class Testwls(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.par = 'B1855+09_NANOGrav_dfg+12_TAI_FB90.par'
        self.tim = 'B1855+09_NANOGrav_dfg+12.tim'
        self.m = mb.get_model(self.par)
        self.t = toa.get_TOAs(self.tim, ephem='DE405')
        self.f = WlsFitter(self.t, self.m)
        # set perturb parameter step
        self.per_param = {'A1': 1e-05, 'DECJ': 1e-06, 'DMX_0003': 120, 'ECC': 0.2,
                          'F0': 1e-12, 'F1': 0.001, 'JUMP3': 10.0, 'M2': 10.0,
                          'OM': 1e-06, 'PB': 1e-08, 'PMDEC': 0.1, 'PMRA': 0.1,
                          'PX': 100, 'RAJ': 1e-08, 'SINI': -0.004075, 'T0': 1e-10}

    def perturb_param(self, param, h):
        self.f.reset_model()
        par = getattr(self.f.model, param)
        orv = par.value
        par.value = (1+h)*orv
        self.f.update_resids()
        self.f.set_fitparams(param)

    def test_wlf_fitter(self):
        for ii, p in enumerate(self.per_param.keys()):
            self.perturb_param(p, self.per_param[p])
            self.f.fit_toas()
            chi2_red = self.f.resids.chi2_reduced
            tol = 2.6
            msg = "Fitting parameter " + p + " failed. with chi2_red " + str(chi2_red)
            assert chi2_red < tol, msg
