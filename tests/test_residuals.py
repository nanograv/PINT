""" Test for residual class
"""

import os
from copy import deepcopy
from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from astropy.time import TimeDelta
from numpy.testing import assert_allclose
from pinttestdata import datadir

from pint.fitter import GLSFitter, WidebandTOAFitter, WLSFitter
from pint.models import get_model
from pint.models.dispersion_model import Dispersion
from pint.residuals import CombinedResiduals, Residuals, WidebandTOAResiduals
from pint.toa import get_TOAs
from pint.simulation import make_fake_toas_uniform
from pint.utils import weighted_mean

os.chdir(datadir)


@pytest.fixture
def wideband_fake():
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            ECORR mjd 57000 58000 2
            """
        )
    )
    toas = make_fake_toas_uniform(
        57000, 59000, 40, model=model, error=1 * u.us, dm=10 * u.pc / u.cm ** 3
    )
    toas.compute_pulse_numbers(model)
    np.random.seed(0)
    toas.adjust_TOAs(TimeDelta(np.random.randn(len(toas)) * u.us))
    for f in toas.table["flags"]:
        f["pp_dm"] = str(float(f["pp_dm"]) + np.random.randn() * float(f["pp_dme"]))
    return toas, model


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
    toas = make_fake_toas_uniform(57000, 59000, 20, model=model, error=1 * u.us)
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
    toas = make_fake_toas_uniform(
        57000, 59000, 20, model=model, error=1 * u.us, dm=10 * u.pc / u.cm ** 3
    )
    r = WidebandTOAResiduals(toas, model)
    e = r.toa.get_data_error(scaled=True)
    assert np.all(e != 0)
    assert 0 < np.sum(e > 1.5 * u.us) < len(toas)
    with pytest.raises(ValueError):
        model.as_parfile().index("EQUAD")


def test_residuals_wls_chi2():
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            """
        )
    )
    toas = make_fake_toas_uniform(57000, 59000, 20, model=model, error=1 * u.us)
    np.random.seed(0)
    toas.adjust_TOAs(TimeDelta(np.random.randn(len(toas)) * u.us))
    r = Residuals(toas, model)
    f = WLSFitter(toas, model)
    assert f.fit_toas() == r.chi2


def test_residuals_gls_chi2():
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            ECORR mjd 57000 58000 2
            """
        )
    )
    toas = make_fake_toas_uniform(57000, 59000, 20, model=model, error=1 * u.us)
    np.random.seed(0)
    toas.adjust_TOAs(TimeDelta(np.random.randn(len(toas)) * u.us))
    r = Residuals(toas, model)
    f = GLSFitter(toas, model)
    assert f.fit_toas() == r.chi2


def test_residuals_wideband_chi2(wideband_fake):
    toas, model = wideband_fake
    r = WidebandTOAResiduals(toas, model)
    rn = Residuals(toas, model)
    f = WidebandTOAFitter(toas, model)
    assert_allclose(f.fit_toas(), r.chi2)
    assert f.fit_toas() >= rn.chi2


# @pytest.mark.xfail()
@pytest.mark.parametrize(
    "full_cov", [pytest.param(True, marks=pytest.mark.xfail), False]
)
def test_gls_chi2_reasonable(full_cov):
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            TNRedAmp -14.227505410948254
            TNRedGam 4.91353
            TNRedC 45
            """
        )
    )
    toas = make_fake_toas_uniform(57000, 59000, 40, model=model, error=1 * u.us)
    np.random.seed(0)
    toas.adjust_TOAs(TimeDelta(np.random.randn(len(toas)) * u.us))
    f = GLSFitter(toas, model)
    fit_chi2 = f.fit_toas(full_cov=full_cov)
    assert_allclose(fit_chi2, f.resids.calc_chi2(full_cov=full_cov))


@pytest.mark.xfail(reason="numerical instability maybe?")
def test_gls_chi2_full_cov():
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            TNRedAmp -14.227505410948254
            TNRedGam 4.91353
            TNRedC 45
            """
        )
    )
    model.free_params = ["ELAT", "ELONG"]
    toas = make_fake_toas_uniform(57000, 59000, 100, model=model, error=1 * u.us)
    np.random.seed(0)
    toas.adjust_TOAs(TimeDelta(np.random.randn(len(toas)) * u.us))
    r = Residuals(toas, model)
    assert_allclose(r.calc_chi2(full_cov=True), r.calc_chi2(full_cov=False))


def test_gls_chi2_behaviour():
    model = get_model(
        StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            TNRedAmp -14.227505410948254
            TNRedGam 4.91353
            TNRedC 45
            """
        )
    )
    model.free_params = ["F0", "ELAT", "ELONG"]
    toas = make_fake_toas_uniform(57000, 59000, 40, model=model, error=1 * u.us)
    np.random.seed(0)
    toas.adjust_TOAs(TimeDelta(np.random.randn(len(toas)) * u.us))
    f = GLSFitter(toas, model)
    initial_chi2 = Residuals(toas, model).calc_chi2()
    fit_chi2 = f.fit_toas()
    assert fit_chi2 <= initial_chi2
    assert f.resids.calc_chi2() <= initial_chi2
    assert initial_chi2 == Residuals(toas, model).calc_chi2()


def test_wideband_chi2_null_updating(wideband_fake):
    toas, model = wideband_fake
    model.free_params = ["F0"]
    f = WidebandTOAFitter(toas, model)
    assert abs(f.fit_toas() - WidebandTOAResiduals(toas, model).chi2) > 1
    c2 = WidebandTOAResiduals(toas, f.model).chi2
    assert_allclose(f.fit_toas(), c2)
    c2 = WidebandTOAResiduals(toas, f.model).chi2
    assert_allclose(f.fit_toas(), c2)


def test_wideband_chi2_updating(wideband_fake):
    toas, model = wideband_fake
    model.free_params = ["F0"]
    model.F0.value += 1e-6
    c2 = WidebandTOAResiduals(
        toas, model, toa_resid_args=dict(track_mode="use_pulse_numbers")
    ).chi2
    f2 = WidebandTOAFitter(
        toas, model, additional_args=dict(toa=dict(track_mode="use_pulse_numbers"))
    )
    ftc2 = f2.fit_toas()
    assert abs(ftc2 - c2) > 100
    assert_allclose(f2.model.F0.value, 1)
    assert 1e-3 > abs(WidebandTOAResiduals(toas, f2.model).chi2 - ftc2) > 1e-5
    ftc2 = f2.fit_toas(maxiter=10)
    assert_allclose(WidebandTOAResiduals(toas, f2.model).chi2, ftc2)
