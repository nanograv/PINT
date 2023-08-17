import os
import pytest

from pint.models.model_builder import get_model
from pint import toa
from pint.fitter import WLSFitter
from pinttestdata import datadir


class Testwls:
    """Compare delays from the dd model with tempo and PINT"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.par = "B1855+09_NANOGrav_dfg+12_TAI_FB90.par"
        cls.tim = "B1855+09_NANOGrav_dfg+12.tim"
        cls.m = get_model(cls.par)
        cls.t = toa.get_TOAs(cls.tim, ephem="DE405")
        cls.f = WLSFitter(cls.t, cls.m)
        # set perturb parameter step
        cls.per_param = {
            "A1": 1e-05,
            "DECJ": 1e-06,
            "DMX_0003": 120,
            "ECC": 0.2,
            "F0": 1e-12,
            "F1": 0.001,
            "JUMP3": 10.0,
            "M2": 10.0,
            "OM": 1e-06,
            "PB": 1e-08,
            "PMDEC": 0.1,
            "PMRA": 0.1,
            "PX": 100,
            "RAJ": 1e-08,
            "SINI": -0.004075,
            "T0": 1e-10,
        }

    def perturb_param(self, param, h):
        self.f.reset_model()
        par = getattr(self.f.model, param)
        orv = par.value
        par.value = (1 + h) * orv
        self.f.update_resids()
        self.f.model.free_params = [param]

    def test_wlf_fitter(self):
        for p in self.per_param.keys():
            self.perturb_param(p, self.per_param[p])
            self.f.fit_toas()
            red_chi2 = self.f.resids.reduced_chi2
            tol = 2.6
            msg = f"Fitting parameter {p} failed. with red_chi2 {str(red_chi2)}"
            assert red_chi2 < tol, msg

    def test_has_correlated_errors(self):
        assert not self.f.resids.model.has_correlated_errors
