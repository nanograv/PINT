""" Test for residual class
"""

import os
import pytest
import numpy as np

import astropy.units as u

from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals, CombinedResiduals
from pint.utils import weighted_mean
from pinttestdata import datadir

os.chdir(datadir)


class TestResidualBuilding:
    def setup(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        self.toa = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")

    def test_build_phase_residual(self):
        phase_res = Residuals(toas=self.toa, model=self.model)
        assert len(phase_res.phase_resids) == self.toa.ntoas

        # Test no mean subtraction
        phase_res_nomean = Residuals(
            toas=self.toa, model=self.model, residual_type="toa", subtract_mean=False,
        )
        assert len(phase_res_nomean.resids) == self.toa.ntoas
        assert phase_res.resids.unit == phase_res.unit
        phase_res_noweight = Residuals(
            toas=self.toa,
            model=self.model,
            residual_type="toa",
            subtract_mean=True,
            use_weighted_mean=False,
        )
        phase_res_no_f0_scale = Residuals(
            toas=self.toa, model=self.model, residual_type="toa", scaled_by_F0=False
        )
        assert phase_res_no_f0_scale.resids.unit == u.Unit("")
        # assert np.all(phase_res_nomean.resids -
        #         phase_res_nomean.resids.mean() == phase_res_noweight.resids)

    def test_build_dm_residual(self):
        dm_res = Residuals(toas=self.toa, model=self.model, residual_type="dm")
        assert len(dm_res.resids) == self.toa.ntoas

        # Test no mean subtraction
        dm_res_nomean = Residuals(
            toas=self.toa, model=self.model, residual_type="dm", subtract_mean=False,
        )
        assert len(dm_res_nomean.resids) == self.toa.ntoas
        weight = 1.0 / (dm_res_nomean.dm_error ** 2)
        wm = (dm_res_nomean.resids * weight).sum() / weight.sum()
        assert np.all(dm_res_nomean.resids - wm == dm_res.resids)
        dm_res_noweight = Residuals(
            toas=self.toa,
            model=self.model,
            residual_type="dm",
            subtract_mean=True,
            use_weighted_mean=False,
        )
        assert np.all(
            dm_res_nomean.resids - dm_res_nomean.resids.mean() == dm_res_noweight.resids
        )

    def test_combined_residuals(self):
        phase_res = Residuals(toas=self.toa, model=self.model)
        dm_res = Residuals(toas=self.toa, model=self.model, residual_type="dm")
        cb_residuals = CombinedResiduals([phase_res, dm_res])
        cb_chi2 = cb_residuals.chi2

        assert len(cb_residuals._combined_resids) == 2 * self.toa.ntoas
        assert cb_residuals.unit["toa"] == u.s
        assert cb_residuals.unit["dm"] == u.pc / u.cm ** 3
        assert cb_chi2 == phase_res.chi2 + dm_res.chi2
