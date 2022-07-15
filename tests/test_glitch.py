#! /usr/bin/env python
import os
import unittest

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

import pint.fitter
import pint.models
import pint.residuals
import pint.toa

parfile = os.path.join(datadir, "prefixtest.par")
timfile = os.path.join(datadir, "prefixtest.tim")


class TestGlitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = pint.models.get_model(parfile)
        cls.t = pint.toa.get_TOAs(timfile, ephem="DE405", include_bipm=False)
        cls.f = pint.fitter.WLSFitter(cls.t, cls.m)

    @pytest.mark.skipif(
        "TEMPO2" not in os.environ,
        reason="Needs TEMPO2 clock files, but TEMPO2 envariable not set",
    )
    def test_glitch(self):
        print("Test prefix parameter via a glitch model")
        rs = pint.residuals.Residuals(self.t, self.m).phase_resids
        # Now do the fit
        print("Fitting...")
        self.f.fit_toas()
        emsg = "RMS of " + self.m.PSR.value + " is too big."
        assert self.f.resids.time_resids.std().to(u.us).value < 950.0, emsg

    @pytest.mark.skipif(
        "TEMPO2" not in os.environ,
        reason="Needs TEMPO2 clock files, but TEMPO2 envariable not set",
    )
    @pytest.mark.filterwarnings("ignore:invalid value")
    def test_glitch_der(self):
        delay = self.m.delay(self.t)
        for pf in self.m.glitch_prop:
            for idx in set(self.m.glitch_indices):
                if pf in ["GLF0D_", "GLTD_"]:
                    getattr(self.m, "GLF0D_%d" % idx).value = 1.0
                    getattr(self.m, "GLTD_%d" % idx).value = 100
                else:
                    getattr(self.m, "GLF0D_%d" % idx).value = 0.0
                    getattr(self.m, "GLTD_%d" % idx).value = 0.0
                param = pf + str(idx)
                adf = self.m.d_phase_d_param(self.t, delay, param)
                param_obj = getattr(self.m, param)
                # Get numerical derivative steps.
                if param_obj.units == u.day:
                    h = 1e-8
                else:
                    h = 1e-2
                ndf = self.m.d_phase_d_param_num(self.t, param, h)
                diff = adf - ndf
                mean = (adf + ndf) / 2.0
                r_diff = diff / mean
                errormsg = (
                    "Derivatives for %s is not accurate, max relative difference is"
                    % param
                )
                errormsg += " %lf" % np.nanmax(np.abs(r_diff.value))
                assert np.nanmax(np.abs(r_diff.value)) < 1e-3, errormsg


if __name__ == "__main__":
    pass
