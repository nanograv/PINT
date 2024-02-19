import json
import os
import pytest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
from pint import toa
from pint.fitter import GLSFitter
from pinttestdata import datadir


class TestGLS:
    """Compare delays from the dd model with tempo and PINT"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.par = "B1855+09_NANOGrav_9yv1.gls.par"
        cls.tim = "B1855+09_NANOGrav_9yv1.tim"
        cls.m = mb.get_model(cls.par)
        cls.t = toa.get_TOAs(cls.tim, ephem="DE436")
        cls.f = GLSFitter(cls.t, cls.m)
        # get tempo2 parameter dict
        with open("B1855+09_tempo2_gls_pars.json", "r") as fp:
            cls.t2d = json.load(fp)
        # Get tempo whitened resids
        mjd, cls.twres = (
            np.genfromtxt("B1855+09_NANOGrav_9yv1_whitened.tempo_test", unpack=True)
            * u.us
        )

    def fit(self, full_cov, debug=False):
        self.f.reset_model()
        self.f.update_resids()
        self.f.fit_toas(full_cov=full_cov, debug=debug)

    def test_gls_fitter(self):
        for full_cov in [True, False]:
            self.fit(full_cov, True)
            for par, val in sorted(self.t2d.items()):
                if par not in ["F0"]:
                    v = (
                        getattr(self.f.model, par).value
                        if par not in ["ELONG", "ELAT"]
                        else getattr(self.f.model, par).quantity.to(u.rad).value
                    )

                    e = (
                        getattr(self.f.model, par).uncertainty.value
                        if par not in ["ELONG", "ELAT"]
                        else getattr(self.f.model, par).uncertainty.to(u.rad).value
                    )
                    msg = f"Parameter {par} does not match T2 for full_cov={full_cov}"
                    assert np.abs(v - val[0]) <= val[1], msg
                    assert np.abs(v - val[0]) <= e, msg
                    assert np.abs(1 - val[1] / e) < 0.1, msg

    def test_noise_design_matrix_index(self):
        self.fit(False, True)  # get the debug info
        # Test red noise basis
        pl_rd = self.f.model.pl_rn_basis_weight_pair(self.f.toas)[0]
        p0, p1 = self.f.resids.pl_red_noise_M_index
        pl_rd_backwards = (
            self.f.resids.pl_red_noise_M[0] * self.f.resids.norm[p0:p1][np.newaxis, :]
        )
        assert np.all(np.isclose(pl_rd, pl_rd_backwards))
        # Test ecorr basis
        ec = self.f.model.ecorr_basis_weight_pair(self.f.toas)[0]
        p0, p1 = self.f.resids.ecorr_noise_M_index
        ec_backwards = (
            self.f.resids.ecorr_noise_M[0] * self.f.resids.norm[p0:p1][np.newaxis, :]
        )
        assert np.all(np.isclose(ec, ec_backwards))

    def test_whitening(self):
        self.fit(full_cov=False)
        wres = self.f.resids.time_resids - self.f.resids.noise_resids["pl_red_noise"]
        wres_diff = wres - self.twres
        wres_diff -= wres_diff.mean()
        assert wres_diff.std() < 10.0 * u.ns
        assert np.abs(wres_diff).max() < 50.0 * u.ns

    def test_gls_compare(self):
        self.fit(full_cov=False)
        chi21 = self.f.resids.chi2
        self.fit(full_cov=True)
        chi22 = self.f.resids.chi2
        assert np.allclose(chi21, chi22)

    def test_has_correlated_errors(self):
        assert self.f.resids.model.has_correlated_errors
