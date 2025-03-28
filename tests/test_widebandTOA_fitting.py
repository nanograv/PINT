""" Various of tests for the general data fitter using wideband TOAs."""

import os

import numpy as np
import astropy.units as u
from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WidebandTOAFitter
from pinttestdata import datadir

os.chdir(datadir)


class TestWidebandTOAFitter:
    def setup_method(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        self.toas = get_TOAs(
            "J1614-2230_NANOGrav_12yv3.wb.tim", ephem="DE436", bipm_version="BIPM2015"
        )
        self.fit_data_name = ["toa", "dm"]
        self.fit_params_lite = ["F0", "F1", "ELONG", "ELAT", "DMJUMP1", "DMX_0022"]
        self.tempo_res = np.genfromtxt(
            "J1614-2230_NANOGrav_12yv3.wb.tempo_test", comments="#"
        )

    def test_fitter_init(self):
        fitter = WidebandTOAFitter([self.toas], self.model, additional_args={})

        # test making residuals
        assert len(fitter.resids._combined_resids) == 2 * self.toas.ntoas
        # test additional args
        add_args = {"toa": {"subtract_mean": False}}
        fitter2 = WidebandTOAFitter([self.toas], self.model, additional_args=add_args)

        assert fitter2.resids.toa.subtract_mean == False

    def test_fitter_designmatrix(self):
        fitter = WidebandTOAFitter([self.toas], self.model, additional_args={})
        fitter.model.free_params = self.fit_params_lite
        assert set(fitter.model.free_params) == set(self.fit_params_lite)
        # test making design matrix
        d_matrix = fitter.get_designmatrix()
        assert d_matrix.shape == (2 * self.toas.ntoas, len(self.fit_params_lite) + 1)
        assert [lb[0] for lb in d_matrix.labels[0]] == ["toa", "dm"]
        assert d_matrix.derivative_params == (["Offset"] + fitter.model.free_params)

    def test_fitting_no_full_cov(self):
        fitter = WidebandTOAFitter([self.toas], self.model, additional_args={})
        time_rms_pre = fitter.resids_init.toa.rms_weighted()
        dm_rms_pre = fitter.resids_init.dm.rms_weighted()
        fitter.fit_toas()
        dm_rms_post = fitter.resids.dm.rms_weighted()

        prefit_pint = fitter.resids_init.toa.time_resids
        prefit_tempo = self.tempo_res[:, 0] * u.us
        diff_prefit = (prefit_pint - prefit_tempo).to(u.ns)
        # 50 ns is the difference of PINT tempo precession and nutation model.
        assert np.abs(diff_prefit - diff_prefit.mean()).max() < 50 * u.ns
        postfit_pint = fitter.resids.toa.time_resids
        postfit_tempo = self.tempo_res[:, 1] * u.us
        diff_postfit = (postfit_pint - postfit_tempo).to(u.ns)
        assert np.abs(diff_postfit - diff_postfit.mean()).max() < 50 * u.ns
        assert np.abs(dm_rms_pre - dm_rms_post) < 5e-8 * dm_rms_pre.unit

    def test_fitting_full_cov(self):
        fitter2 = WidebandTOAFitter([self.toas], self.model, additional_args={})
        time_rms_pre = fitter2.resids_init.toa.rms_weighted()
        dm_rms_pre = fitter2.resids_init.dm.rms_weighted()
        fitter2.fit_toas(full_cov=True)
        time_rms_post = fitter2.resids.toa.rms_weighted()
        dm_rms_post = fitter2.resids.dm.rms_weighted()

        prefit_pint = fitter2.resids_init.toa.time_resids
        prefit_tempo = self.tempo_res[:, 0] * u.us
        diff_prefit = (prefit_pint - prefit_tempo).to(u.ns)
        # 50 ns is the difference of PINT tempo precession and nautation model.
        assert np.abs(diff_prefit - diff_prefit.mean()).max() < 50 * u.ns
        postfit_pint = fitter2.resids.toa.time_resids
        postfit_tempo = self.tempo_res[:, 1] * u.us
        diff_postfit = (postfit_pint - postfit_tempo).to(u.ns)
        assert np.abs(diff_postfit - diff_postfit.mean()).max() < 50 * u.ns
        assert np.abs(dm_rms_pre - dm_rms_post) < 5e-8 * dm_rms_pre.unit

    def test_noise_design_matrix_index(self):
        model = get_model("B1855+09_NANOGrav_12yv3.wb.gls.par")
        toas = get_TOAs(
            "B1855+09_NANOGrav_12yv3.wb.tim", ephem="DE436", bipm_version="BIPM2015"
        )
        fitter = WidebandTOAFitter(toas, model, additional_args={})
        fitter.fit_toas(full_cov=False, debug=True)
        # Test red noise basis
        pl_rd = fitter.model.pl_rn_basis_weight_pair(fitter.toas)[0]
        p0, p1 = fitter.resids.pl_red_noise_M_index
        pl_rd_backwards = (
            fitter.resids.pl_red_noise_M[0] * fitter.resids.norm[p0:p1][np.newaxis, :]
        )
        assert np.all(np.isclose(pl_rd, pl_rd_backwards[0:313, :]))
