from pinttestdata import datadir
from pint.models import get_model
from pint.simulation import make_fake_toas_fromtim
from pint.residuals import Residuals
from pint.fitter import DownhillWLSFitter

import pytest
from io import StringIO
import numpy as np
from copy import deepcopy


@pytest.fixture
def model_and_toas():
    model_raw = get_model(datadir / "J1909-3744.NB.par")
    fdlines = (
        "FD1JUMP -sys GM_GWB_500_100_b1 0.01 1\n"
        "FD1JUMP -sys GM_GWB_1460_100_b1 0.02 1\n"
        "FD2JUMP -sys GM_GWB_500_100_b1 0.001 1\n"
        "FD2JUMP -sys GM_GWB_1460_100_b1 0.002 1\n"
        "FDJUMPDM -sys GM_GWB_500_100_b1 0.00002 1\n"
        "FDJUMPDM -sys GM_GWB_1460_100_b1 0.00001 1\n"
    )
    par = str(model_raw) + fdlines

    model = get_model(StringIO(par))
    toas = make_fake_toas_fromtim(datadir / "J1909-3744.NB.tim", model, add_noise=True)

    return model, toas


def test_params(model_and_toas):
    model, toas = model_and_toas

    assert (
        hasattr(model, "FD1JUMP1")
        and hasattr(model, "FD2JUMP1")
        and hasattr(model, "FD1JUMP2")
        and hasattr(model, "FD2JUMP2")
        and hasattr(model, "FDJUMPDM1")
        and hasattr(model, "FDJUMPDM2")
    )


def test_residuals(model_and_toas):
    model, toas = model_and_toas
    res = Residuals(toas, model)

    assert np.isfinite(res.chi2) and res.reduced_chi2 < 1.5


def test_fitting(model_and_toas):
    model, toas = model_and_toas
    ftr = DownhillWLSFitter(toas, model)
    ftr.fit_toas()

    assert np.isfinite(ftr.resids.chi2) and ftr.resids.reduced_chi2 < 1.5


def test_refitting(model_and_toas):
    model, toas = model_and_toas
    FD1JUMP1_value_original = model.FD1JUMP1.value
    model.FD1JUMP1.value = 0
    model.FDJUMPDM1.value = 0

    ftr = DownhillWLSFitter(toas, model)
    ftr.fit_toas()

    assert (
        np.abs(ftr.model.FD1JUMP1.value - FD1JUMP1_value_original)
        / ftr.model.FD1JUMP1.uncertainty_value
        < 2.5
    )


def test_parfile_write(model_and_toas):
    model, toas = model_and_toas

    par = str(model)

    assert par.count("FD1JUMP") == 2 and par.count("FD2JUMP") == 2


def test_tempo2_parfile_read_write():
    model_raw = get_model(datadir / "J1909-3744.NB.par")
    model_raw.PLANET_SHAPIRO.value = False
    fdlines = (
        "FDJUMP1 -sys GM_GWB_500_100_b1 0.01 1\n"
        "FDJUMP1 -sys GM_GWB_1460_100_b1 0.02 1\n"
        "FDJUMP2 -sys GM_GWB_500_100_b1 0.001 1\n"
        "FDJUMP2 -sys GM_GWB_1460_100_b1 0.002 1\n"
    )
    par = str(model_raw) + fdlines

    model = get_model(StringIO(par))

    assert (
        hasattr(model, "FD1JUMP1")
        and hasattr(model, "FD2JUMP1")
        and hasattr(model, "FD1JUMP2")
        and hasattr(model, "FD2JUMP2")
    )

    par = model.as_parfile()
    assert par.count("FD1JUMP") == 2 and par.count("FD2JUMP") == 2
    assert par.count("FDJUMP1") == 0 and par.count("FDJUMP1") == 0

    par = model.as_parfile(format="tempo2")
    assert par.count("FD1JUMP") == 0 and par.count("FD2JUMP") == 0
    assert par.count("FDJUMP1") == 2 and par.count("FDJUMP1") == 2


def test_residual_change(model_and_toas):
    model, toas = model_and_toas

    r_old = Residuals(toas, model, subtract_mean=False).calc_time_resids().value

    mask_FD1JUMP1 = model.FD1JUMP1.select_toa_mask(toas)
    mask_FD1JUMP1_inv = np.setdiff1d(np.arange(len(toas)), mask_FD1JUMP1)

    model.FD1JUMP1.value = 0

    r_new = Residuals(toas, model, subtract_mean=False).calc_time_resids().value

    assert not np.allclose(r_old[mask_FD1JUMP1], r_new[mask_FD1JUMP1])
    assert np.allclose(r_old[mask_FD1JUMP1_inv], r_new[mask_FD1JUMP1_inv])


def test_fdjumpdm_offset(model_and_toas):
    model, toas = model_and_toas

    model2 = deepcopy(model)
    model2.FDJUMPDM1.value = 0

    mask = model.FDJUMPDM1.select_toa_mask(toas)
    not_mask = np.setdiff1d(np.arange(len(toas)), mask)

    dm1 = model.total_dm(toas)
    dm2 = model2.total_dm(toas)

    assert np.allclose((dm1 - dm2)[not_mask], 0)
    assert np.allclose((dm1 - dm2)[mask], -model.FDJUMPDM1.quantity)
