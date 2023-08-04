from pinttestdata import datadir
from pint.models import get_model
from pint.simulation import make_fake_toas_fromtim
from pint.residuals import Residuals
from pint.fitter import DownhillWLSFitter

import pytest
from io import StringIO
import numpy as np


@pytest.fixture
def model_and_toas():
    model_raw = get_model(datadir / "J1909-3744.NB.par")
    model_raw.PLANET_SHAPIRO.value = False
    fdlines = (
        "FD1JUMP -sys GM_GWB_500_100_b1 0.01 1\n"
        "FD1JUMP -sys GM_GWB_1460_100_b1 0.02 1\n"
        "FD2JUMP -sys GM_GWB_500_100_b1 0.001 1\n"
        "FD2JUMP -sys GM_GWB_1460_100_b1 0.002 1\n"
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
