import io
import re
from copy import deepcopy

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import log
from astropy.time import TimeDelta
from scipy.linalg import block_diag, cho_factor, cho_solve, cholesky

import pint.fitter
from pint.models import get_model
from pint.models.timing_model import MissingTOAs
from pint.toa import make_fake_toas, merge_TOAs

par_eccentric = """
PSR J1234+5678
F0 500 0 1
ELAT 0 0 1
ELONG 0 0 1
PEPOCH 57000
POSEPOCH 57000
DM 10 0 1
SOLARN0 0 0 1
BINARY BT
PB 1 0 1
A1 10 0 1
ECC 0.95 0 1
OM 0 0 1
T0 57000 0 1
"""


@pytest.fixture(scope="module")
def model_eccentric_toas():
    g = np.random.default_rng(0)
    model_eccentric = get_model(io.StringIO(par_eccentric))

    toas = make_fake_toas(57000, 57001, 20, model_eccentric, freq=1400, obs="@")
    toas.adjust_TOAs(TimeDelta(g.standard_normal(len(toas)) * toas.table["error"]))

    return model_eccentric, toas


@pytest.fixture(scope="module")
def model_eccentric_toas_ecorr():
    g = np.random.default_rng(0)
    model_eccentric = get_model(
        io.StringIO("\n".join([par_eccentric, "ECORR tel @ 2"]))
    )

    toas = merge_TOAs(
        [
            make_fake_toas(57000, 57001, 20, model_eccentric, freq=1000, obs="@"),
            make_fake_toas(57000, 57001, 20, model_eccentric, freq=2000, obs="@"),
        ]
    )
    toas.adjust_TOAs(TimeDelta(g.standard_normal(len(toas)) * toas.table["error"]))

    return model_eccentric, toas


def test_wls_full_procedure(model_eccentric_toas):
    model_eccentric, toas = model_eccentric_toas
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.DownhillWLSFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]

    f.fit_toas(maxiter=10)

    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4


def test_gls_full_procedure(model_eccentric_toas_ecorr):
    model_eccentric, toas = model_eccentric_toas_ecorr
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.DownhillGLSFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]

    f.fit_toas(maxiter=10)

    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4


def test_wls_two_step(model_eccentric_toas):
    model_eccentric, toas = model_eccentric_toas
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.DownhillWLSFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]
    f.fit_toas(maxiter=2)
    f2 = pint.fitter.DownhillWLSFitter(toas, model_wrong)
    f2.model.free_params = ["ECC"]
    f2.fit_toas(maxiter=1)
    f2.fit_toas(maxiter=1)
    assert np.abs(f.model.ECC.value - f2.model.ECC.value) < 1e-12


def test_gls_two_step(model_eccentric_toas_ecorr):
    model_eccentric, toas = model_eccentric_toas_ecorr
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.DownhillGLSFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]
    f.fit_toas(maxiter=2)
    f2 = pint.fitter.DownhillGLSFitter(toas, model_wrong)
    f2.model.free_params = ["ECC"]
    f2.fit_toas(maxiter=1)
    f2.fit_toas(maxiter=1)
    assert np.abs(f.model.ECC.value - f2.model.ECC.value) < 1e-12


def test_detect_gls_needed(model_eccentric_toas_ecorr):
    model_eccentric, toas = model_eccentric_toas_ecorr
    with pytest.raises(pint.fitter.CorrelatedErrors) as e:
        pint.fitter.DownhillWLSFitter(toas, model_eccentric)
    assert e.value.trouble_components == ["EcorrNoise"]
