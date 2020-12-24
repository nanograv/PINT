""" Test for residual class
"""

import os
from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

from pint.models import get_model
from pint.models.dispersion_model import Dispersion
from pint.residuals import CombinedResiduals, Residuals, WidebandTOAResiduals
from pint.toa import get_TOAs, make_fake_toas
from pint.utils import weighted_mean

os.chdir(datadir)


class TestResidualBuilding:
    def setup(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        # self.toa = make_fake_toas(57000,59000,20,model=self.model)
        self.toa = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")

    def test_build_phase_residual(self):
        phase_res = Residuals(toas=self.toa, model=self.model)
        assert len(phase_res.phase_resids) == self.toa.ntoas

        # Test no mean subtraction
        phase_res_nomean = Residuals(
            toas=self.toa, model=self.model, residual_type="toa", subtract_mean=False
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
        # assert np.all(phase_res_nomean.resids -
        #         phase_res_nomean.resids.mean() == phase_res_noweight.resids)

    def test_build_dm_residual(self):
        dm_res = Residuals(toas=self.toa, model=self.model, residual_type="dm")
        assert len(dm_res.resids) == self.toa.ntoas

        # Test no mean subtraction
        dm_res_nomean = Residuals(
            toas=self.toa, model=self.model, residual_type="dm", subtract_mean=True
        )
        assert len(dm_res_nomean.resids) == self.toa.ntoas
        weight = 1.0 / (dm_res.dm_error ** 2)
        wm = np.average(dm_res.resids, weights=weight)
        assert np.all(dm_res.resids - wm == dm_res_nomean.resids)
        dm_res_noweight = Residuals(
            toas=self.toa,
            model=self.model,
            residual_type="dm",
            subtract_mean=True,
            use_weighted_mean=False,
        )
        assert np.all(dm_res.resids - dm_res.resids.mean() == dm_res_noweight.resids)

    def test_combined_residuals(self):
        phase_res = Residuals(toas=self.toa, model=self.model)
        dm_res = Residuals(toas=self.toa, model=self.model, residual_type="dm")
        cb_residuals = CombinedResiduals([phase_res, dm_res])
        cb_chi2 = cb_residuals.chi2

        assert len(cb_residuals._combined_resids) == 2 * self.toa.ntoas
        assert cb_residuals.unit["toa"] == u.s
        assert cb_residuals.unit["dm"] == u.pc / u.cm ** 3
        assert cb_chi2 == phase_res.chi2 + dm_res.chi2
        with pytest.raises(AttributeError):
            cb_residuals.dof
        with pytest.raises(AttributeError):
            cb_residuals.residual_objs["toa"].dof
        with pytest.raises(AttributeError):
            cb_residuals.residual_objs["dm"].dof
        with pytest.raises(AttributeError):
            cb_residuals.model

    def test_wideband_residuals(self):
        wb_res = WidebandTOAResiduals(toas=self.toa, model=self.model)
        assert wb_res.dof == 419
        # Make sure the model object are shared by all individual residual class
        assert wb_res.model is wb_res.toa.model
        assert wb_res.model is wb_res.dm.model


def test_residuals_scaled_uncertainties():
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            EFAC mjd 57000 58000 2
            """
        )
    )
    toas = make_fake_toas(57000, 59000, 20, model=model, error=1 * u.us)
    r = Residuals(toas, model)
    e = r.get_data_error(scaled=True)
    assert np.all(e != 0)
    assert 0 < np.sum(e > 1.5 * u.us) < len(toas)
    with pytest.raises(ValueError):
        model.as_parfile().index("EQUAD")


def test_residuals_fake_wideband():
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            EFAC mjd 57000 58000 2
            """
        )
    )
    toas = make_fake_toas(57000, 59000, 20, model=model, error=1 * u.us, dm=10)
    r = WidebandTOAResiduals(toas, model)
    e = r.toa.get_data_error(scaled=True)
    assert np.all(e != 0)
    assert 0 < np.sum(e > 1.5 * u.us) < len(toas)
    with pytest.raises(ValueError):
        model.as_parfile().index("EQUAD")
