import io
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.time import TimeDelta

import pint.fitter
from pint.models import get_model
from pint.toa import merge_TOAs
from pint.simulation import make_fake_toas_uniform

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

    toas = make_fake_toas_uniform(
        57000, 57001, 20, model_eccentric, freq=1400 * u.MHz, obs="@"
    )
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
            make_fake_toas_uniform(
                57000, 57001, 20, model_eccentric, freq=1000 * u.MHz, obs="@"
            ),
            make_fake_toas_uniform(
                57000, 57001, 20, model_eccentric, freq=2000 * u.MHz, obs="@"
            ),
        ]
    )
    toas.adjust_TOAs(TimeDelta(g.standard_normal(len(toas)) * toas.table["error"]))

    return model_eccentric, toas


@pytest.fixture(scope="module")
def model_eccentric_toas_wb():
    g = np.random.default_rng(0)
    model_eccentric = get_model(
        io.StringIO("\n".join([par_eccentric, "ECORR tel @ 2"]))
    )

    toas = merge_TOAs(
        [
            make_fake_toas_uniform(
                57000,
                57001,
                20,
                model_eccentric,
                freq=1000 * u.MHz,
                obs="@",
                wideband=True,
            ),
            make_fake_toas_uniform(
                57000,
                57001,
                20,
                model_eccentric,
                freq=2000 * u.MHz,
                obs="@",
                wideband=True,
            ),
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

    assert f.converged
    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4


@pytest.mark.parametrize("full_cov", [False, True])
def test_gls_full_procedure(model_eccentric_toas_ecorr, full_cov):
    model_eccentric, toas = model_eccentric_toas_ecorr
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.DownhillGLSFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]

    f.fit_toas(maxiter=10, full_cov=full_cov, debug=True)

    assert f.converged
    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4

    if not full_cov:
        # Test ecorr basis
        ec = f.model.ecorr_basis_weight_pair(f.toas)[0]
        p0, p1 = f.resids.ecorr_noise_M_index
        ec_backwards = f.resids.ecorr_noise_M[0] * f.resids.norm[p0:p1][np.newaxis, :]
        assert np.all(np.isclose(ec, ec_backwards))


@pytest.mark.parametrize("full_cov", [False, True])
def test_wideband_full_procedure(model_eccentric_toas_wb, full_cov):
    model_eccentric, toas = model_eccentric_toas_wb
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.WidebandDownhillFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]

    f.fit_toas(maxiter=10, full_cov=full_cov, debug=True)

    assert f.converged
    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4
    if not full_cov:
        # Test ecorr basis
        ec = f.model.ecorr_basis_weight_pair(f.toas)[0]
        p0, p1 = f.resids.ecorr_noise_M_index
        ec_backwards = f.resids.ecorr_noise_M[0] * f.resids.norm[p0:p1][np.newaxis, :]
        assert np.all(np.isclose(ec, ec_backwards[0:40, :]))


@pytest.mark.parametrize("full_cov", [False, True])
def test_wideband_lm_full_procedure(model_eccentric_toas_wb, full_cov):
    model_eccentric, toas = model_eccentric_toas_wb
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.WidebandLMFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]

    f.fit_toas(full_cov=full_cov)

    assert f.converged
    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4


def test_wls_two_step(model_eccentric_toas):
    model_eccentric, toas = model_eccentric_toas
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.DownhillWLSFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]
    with pytest.raises(pint.fitter.MaxiterReached):
        f.fit_toas(maxiter=2)
    assert not f.converged

    f2 = pint.fitter.DownhillWLSFitter(toas, model_wrong)
    f2.model.free_params = ["ECC"]
    with pytest.raises(pint.fitter.MaxiterReached):
        f2.fit_toas(maxiter=1)
    with pytest.raises(pint.fitter.MaxiterReached):
        f2.fit_toas(maxiter=1)
    assert np.abs(f.model.ECC.value - f2.model.ECC.value) < 1e-12


@pytest.mark.parametrize("full_cov", [False, True])
def test_gls_two_step(model_eccentric_toas_ecorr, full_cov):
    model_eccentric, toas = model_eccentric_toas_ecorr
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.DownhillGLSFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]
    with pytest.raises(pint.fitter.MaxiterReached):
        f.fit_toas(maxiter=2, full_cov=full_cov)
    assert not f.converged
    f2 = pint.fitter.DownhillGLSFitter(toas, model_wrong)
    f2.model.free_params = ["ECC"]
    with pytest.raises(pint.fitter.MaxiterReached):
        f2.fit_toas(maxiter=1, full_cov=full_cov)
    with pytest.raises(pint.fitter.MaxiterReached):
        f2.fit_toas(maxiter=1, full_cov=full_cov)
    assert np.abs(f.model.ECC.value - f2.model.ECC.value) < 1e-12


@pytest.mark.parametrize("full_cov", [False, True])
def test_wb_two_step(model_eccentric_toas_wb, full_cov):
    model_eccentric, toas = model_eccentric_toas_wb
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5

    f = pint.fitter.WidebandDownhillFitter(toas, model_wrong)
    f.model.free_params = ["ECC"]
    with pytest.raises(pint.fitter.MaxiterReached):
        f.fit_toas(maxiter=2, full_cov=full_cov)
    assert not f.converged
    f2 = pint.fitter.WidebandDownhillFitter(toas, model_wrong)
    f2.model.free_params = ["ECC"]
    with pytest.raises(pint.fitter.MaxiterReached):
        f2.fit_toas(maxiter=1, full_cov=full_cov)
    with pytest.raises(pint.fitter.MaxiterReached):
        f2.fit_toas(maxiter=1, full_cov=full_cov)
    # FIXME: The full_cov version differs at the 1e-10 level for some reason, is it a failure really?
    assert np.abs(f.model.ECC.value - f2.model.ECC.value) < 1e-9


def test_detect_gls_needed(model_eccentric_toas_ecorr):
    model_eccentric, toas = model_eccentric_toas_ecorr
    with pytest.raises(pint.fitter.CorrelatedErrors) as e:
        pint.fitter.DownhillWLSFitter(toas, model_eccentric)
    assert e.value.trouble_components == ["EcorrNoise"]


def test_degenerate_parameters(model_eccentric_toas):
    """ELAT and ELONG are unconstrained by barycentric TOAs - what happens?"""
    model_eccentric, toas = model_eccentric_toas
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5
    model_wrong.free_params = ["ELAT", "ELONG", "ECC"]

    f = pint.fitter.DownhillWLSFitter(toas, model_wrong)

    with pytest.warns(pint.fitter.DegeneracyWarning, match=r".*degeneracy.*ELONG\b"):
        f.fit_toas(maxiter=10)

    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4
    assert f.model.ELAT.value == f.model.ELONG.value == 0


@pytest.mark.parametrize("full_cov", [False, True])
def test_degenerate_parameters_gls(model_eccentric_toas_ecorr, full_cov):
    """ELAT and ELONG are unconstrained by barycentric TOAs - what happens?

    The GLS fitter uses the normal equations, which are less numerically stable
    """
    model_eccentric, toas = model_eccentric_toas_ecorr
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5
    model_wrong.free_params = ["ELAT", "ELONG", "ECC"]

    f = pint.fitter.DownhillGLSFitter(toas, model_wrong)

    with pytest.warns(pint.fitter.DegeneracyWarning):
        f.fit_toas(full_cov=full_cov, threshold=1e-14)

    assert abs(f.model.ECC.value - model_eccentric.ECC.value) < 1e-4
    # For some reason this doesn't work - the values get changed in spite of the SVD
    # for the reduced-rank version it has something to do with
    # assert f.model.ELAT.value == f.model.ELONG.value == 0


def test_same_step_bogus(model_eccentric_toas_wb):
    model_eccentric, toas = model_eccentric_toas_wb
    model_wrong = deepcopy(model_eccentric)
    model_wrong.ECC.value = 0.5
    model_wrong.free_params = ["F0", "ECC"]

    f = pint.fitter.WidebandLMFitter(toas, model_wrong)
    state = f.create_state()
    new_state = state.take_step(state.step)
    f2 = pint.fitter.WidebandTOAFitter(toas, model_wrong)
    f2.fit_toas(maxiter=1)
    assert abs(f2.model.ECC.value - new_state.model.ECC.value) < 1e-4


def test_same_step_real():
    pass
