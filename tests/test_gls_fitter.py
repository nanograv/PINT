#! /usr/bin/env python
import time, sys, os, unittest
import pint.models.model_builder as mb
from pint.phase import Phase
from pint import toa
from pint.fitter import WlsFitter, GLSFitter
import numpy as np
import astropy.units as u
import json

from pinttestdata import testdir, datadir

os.chdir(datadir)
class TestGLS(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.par = 'B1855+09_NANOGrav_9yv1.gls.par'
        self.tim = 'B1855+09_NANOGrav_9yv1.tim'
        self.m = mb.get_model(self.par)
        self.t = toa.get_TOAs(self.tim, ephem='DE436')
        self.f = GLSFitter(self.t, self.m)
        # get tempo2 parameter dict
        with open('B1855+09_tempo2_gls_pars.json', 'r') as fp:
            self.t2d = json.load(fp)

    def fit(self, full_cov):
        self.f.reset_model()
        self.f.update_resids()
        self.f.fit_toas(full_cov=full_cov)

    def test_gls_fitter(self):
        for full_cov in [True, False]:
            self.fit(full_cov)
            for par, val in sorted(self.t2d.items()):
                if par not in ['F0']:
                    v = (getattr(self.f.model, par).value if par not in
                         ['ELONG', 'ELAT'] else
                         getattr(self.f.model, par).quantity.to(u.rad).value)

                    e = (getattr(self.f.model, par).uncertainty.value if par not
                         in ['ELONG', 'ELAT'] else
                         getattr(self.f.model, par).uncertainty.to(u.rad).value)
                    msg = 'Parameter {} does not match T2 for full_cov={}'.format(par, full_cov)
                    assert np.abs(v-val[0]) <= val[1], msg
                    assert np.abs(v-val[0]) <= e, msg
                    assert np.abs(1 - val[1]/e) < 0.1, msg

    def test_gls_compare(self):
        self.fit(full_cov=False)
        chi21 = self.f.resids.chi2
        self.fit(full_cov=True)
        chi22 = self.f.resids.chi2
        assert np.allclose(chi21, chi22)
