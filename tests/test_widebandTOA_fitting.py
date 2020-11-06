""" Various of tests for the general data fitter using wideband TOAs.
"""

import pytest
import os

import numpy as np
import astropy.units as u
from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WidebandTOAFitter
from pinttestdata import datadir

os.chdir(datadir)


class TestWidebandTOAFitter:
    def setup(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        self.toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")
        self.fit_data_name = ["toa", "dm"]
        self.fit_params_lite = ["F0", "F1", "ELONG", "ELAT", "DMJUMP1", "DMX_0022"]
        self.tempo_res = np.genfromtxt('J1614-2230_NANOGrav_12yv3.wb.tempo_test', comments='#')        

    def test_fitter_init(self):
        fitter = WidebandTOAFitter([self.toas,], self.model, additional_args={})

        # test making residuals
        assert len(fitter.resids._combined_resids) == 2 * self.toas.ntoas
        # test additional args
        add_args = {}
        add_args["toa"] = {"subtract_mean": False}
        fitter2 = WidebandTOAFitter([self.toas,], self.model, additional_args=add_args)
        
        assert fitter2.resids.residual_objs['toa'].subtract_mean == False

    def test_fitter_designmatrix(self):
        fitter = WidebandTOAFitter([self.toas,], self.model, additional_args={})
        fitter.set_fitparams(self.fit_params_lite)
        assert set(fitter.get_fitparams()) == set(self.fit_params_lite)
        # test making design matrix
        d_matrix = fitter.get_designmatrix()
        assert d_matrix.shape == (2 * self.toas.ntoas, len(self.fit_params_lite) + 1)
        assert [lb[0] for lb in d_matrix.labels[0]] == ["toa", "dm"]
        assert d_matrix.derivative_params == (
            ["Offset"] + list(fitter.get_fitparams().keys())
        )

    def test_fitting_no_full_cov(self):
        fitter = WidebandTOAFitter([self.toas,], self.model, additional_args={})
        time_rms_pre = fitter.resids_init.residual_objs['toa'].rms_weighted()
        dm_rms_pre = fitter.resids_init.residual_objs['dm'].rms_weighted()
        fitter.fit_toas()
        time_rms_post = fitter.resids.residual_objs['toa'].rms_weighted()
        dm_rms_post = fitter.resids.residual_objs['dm'].rms_weighted()
        # The tolarance may need to be changed
        assert np.abs(time_rms_pre - time_rms_post) < 2e-9 * u.s
        assert np.abs(dm_rms_pre - dm_rms_post) < 3e-8 * u.s

    def test_fitting_full_cov(self):
        fitter = WidebandTOAFitter([self.toas,], self.model, additional_args={})
        time_rms_pre = fitter.resids_init.residual_objs['toa'].rms_weighted()
        dm_rms_pre = fitter.resids_init.residual_objs['dm'].rms_weighted()
        fitter.fit_toas(full_cov=True)
        time_rms_post = fitter.resids.residual_objs['toa'].rms_weighted()
        dm_rms_post = fitter.resids.residual_objs['dm'].rms_weighted()
        # The tolarance may need to be changed
        assert np.abs(time_rms_pre - time_rms_post) < 2e-9 * u.s
        assert np.abs(dm_rms_pre - dm_rms_post) < 3e-8 * u.s
