#! /usr/bin/env python
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import matplotlib.pyplot as plt
import astropy.units as u
import sys
import os
import unittest
import numpy as np
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'prefixtest.par')
timfile = os.path.join(datadir, 'prefixtest.tim')


class TestGlitch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = pint.models.get_model(parfile)
        self.t = pint.toa.get_TOAs(timfile,ephem="DE405", include_bipm=False)
        self.f = pint.fitter.PowellFitter(self.t, self.m)
    def test_glitch(self):
        print("Test prefix parameter via a glitch model")
        rs = pint.residuals.resids(self.t, self.m).phase_resids
        # Now do the fit
        print("Fitting...")
        self.f.fit_toas()
        emsg = "RMS of " + self.m.PSR.value + " is too big."
        assert self.f.resids.time_resids.std().to(u.us).value < 950.0, emsg
        delay = self.m.delay(self.t.table)
        for pf in self.m.glitch_prop:
            for idx in set(self.m.glitch_indices):
                if pf in ['GLF0D_', 'GLTD_']:
                    getattr(self.m, 'GLF0D_%d' % idx ).value = 1.0
                    getattr(self.m , 'GLTD_%d' % idx ).value = 100
                else:
                    getattr(self.m, 'GLF0D_%d' % idx ).value = 0.0
                    getattr(self.m , 'GLTD_%d' % idx ).value = 0.0
                param = pf+str(idx)
                adf = self.m.d_phase_d_param(self.t.table, delay, param)
                ndf = self.m.d_phase_d_param_num(self.t.table, param)
                diff = adf - ndf
                mean = (adf + ndf)/2.0
                r_diff = diff/mean
                errormsg = "Derivatives for %s is not accurate, max relative difference is" % param
                errormsg += " %lf" % np.nanmax(np.abs(r_diff.value))
                assert np.nanmax(np.abs(r_diff.value)) < 1e-3, errormsg



if __name__ == '__main__':
    pass
