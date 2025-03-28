"""Tests for the bayesian interface."""

import io
import pytest
import numpy as np

from scipy.stats import uniform

from pint.bayesian import BayesianTiming
from pint.models import get_model_and_toas, get_model
from pint.config import examplefile
from pint.models.priors import Prior


def set_dummy_priors(model):
    for par in model.free_params:
        param = getattr(model, par)
        param_min = float(param.value - 10 * param.uncertainty_value)
        param_span = float(20 * param.uncertainty_value)
        param.prior = Prior(uniform(param_min, param_span))


@pytest.fixture()
def data_NGC6440E():
    parfile = examplefile("NGC6440E.par.good")
    timfile = examplefile("NGC6440E.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    set_dummy_priors(model)
    return model, toas


@pytest.fixture()
def data_NGC6440E_default_priors():
    parfile = examplefile("NGC6440E.par.good")
    timfile = examplefile("NGC6440E.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    return model, toas


@pytest.fixture()
def data_NGC6440E_efac():
    parfile = examplefile("NGC6440E.par.good")
    timfile = examplefile("NGC6440E.tim")
    model, toas = get_model_and_toas(parfile, timfile)

    parfile = f"{str(model)}EFAC TEL gbt 1 1"
    model = get_model(io.StringIO(parfile))
    set_dummy_priors(model)

    model.EFAC1.prior = Prior(uniform(0.1, 1.9))

    return model, toas


@pytest.fixture()
def data_J0740p6620_wb():
    parfile = examplefile("J0740+6620.FCP+21.wb.DMX3.0.par")
    timfile = examplefile("J0740+6620.FCP+21.wb.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    set_dummy_priors(model)
    return model, toas


@pytest.fixture()
def data_NGC6440E_red():
    parfile = examplefile("NGC6440E.par.good")
    timfile = examplefile("NGC6440E.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    parfile = (
        str(model)
        + """RNAMP         1e-6
             RNIDX         -3.82398"""
    )
    model = get_model(io.StringIO(parfile))
    set_dummy_priors(model)

    model.RNAMP.prior = Prior(uniform(1e-8, 1e-5))
    model.RNAMP.prior = Prior(uniform(-6, -1))

    return model, toas


def test_use_pulse_numbers(data_NGC6440E):
    model, toas = data_NGC6440E
    toas.compute_pulse_numbers(model)
    bt = BayesianTiming(model, toas, use_pulse_numbers=True)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl = bt.lnlikelihood(maxlike_params)
    assert not np.isnan(lnl)


def test_no_noise(data_NGC6440E):
    model, toas = data_NGC6440E
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl = bt.lnlikelihood(maxlike_params)
    assert bt.likelihood_method == "wls" and not np.isnan(lnl)


def test_white_noise(data_NGC6440E_efac):
    model, toas = data_NGC6440E_efac
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl = bt.lnlikelihood(maxlike_params)
    assert bt.likelihood_method == "wls" and not np.isnan(lnl)


def test_lnlikelihood_unit_efac(data_NGC6440E, data_NGC6440E_efac):
    """Log likelihood with no EFAC should be equal to that with EFAC=1."""
    model, toas = data_NGC6440E
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl1 = bt.lnlikelihood(maxlike_params)

    model, toas = data_NGC6440E_efac
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl2 = bt.lnlikelihood(maxlike_params)

    assert np.isclose(lnl1, lnl2)


def test_bayesian_timing_funcs(data_NGC6440E_efac):
    """Test if the prior, likelihood and posterior functions work."""
    model, toas = data_NGC6440E_efac

    bt = BayesianTiming(model, toas)

    nparams = bt.nparams
    assert nparams == len(model.free_params)

    test_cube = 0.5 * np.ones(nparams)
    test_params = bt.prior_transform(test_cube)
    assert np.all(np.isfinite(test_params))

    lnpr = bt.lnprior(test_params)
    assert np.isfinite(lnpr)

    lnl = bt.lnlikelihood(test_params)
    assert np.isfinite(lnl)

    lnp = bt.lnposterior(test_params)
    assert np.isfinite(lnp) and np.isclose(lnp, lnpr + lnl)

    # parameters outside prior range
    test_cube = np.ones(nparams)
    test_params = bt.prior_transform(test_cube) + np.ones_like(test_cube)
    assert bt.lnprior(test_params) == -np.inf
    assert bt.lnposterior(test_params) == -np.inf

    # wrong number of parameters
    with pytest.raises(IndexError):
        bt.lnprior([1])


def test_prior_dict(data_NGC6440E_efac):
    model, toas = data_NGC6440E_efac

    prior_info = {}
    for par in model.free_params:
        param = getattr(model, par)
        param_min = float(param.value - 10 * param.uncertainty_value)
        param_max = float(param.value + 10 * param.uncertainty_value)
        prior_info[par] = {"distr": "uniform", "pmin": param_min, "pmax": param_max}
    prior_info["EFAC1"] = {"distr": "normal", "mu": 1, "sigma": 0.1}

    bt = BayesianTiming(model, toas, use_pulse_numbers=True, prior_info=prior_info)

    test_cube = 0.5 * np.ones(bt.nparams)
    test_params = bt.prior_transform(test_cube)
    assert np.all(np.isfinite(test_params))

    lnpr = bt.lnprior(test_params)
    assert np.isfinite(lnpr)

    with pytest.raises(NotImplementedError):
        prior_info["EFAC1"] = {"distr": "loguniform", "pmin": 0.1, "pmax": 2.0}
        bt = BayesianTiming(model, toas, use_pulse_numbers=True, prior_info=prior_info)


def test_wideband_data(data_J0740p6620_wb):
    model, toas = data_J0740p6620_wb
    bt = BayesianTiming(model, toas)

    assert bt.is_wideband and bt.likelihood_method == "wls"

    test_cube = 0.5 * np.ones(bt.nparams)
    test_params = bt.prior_transform(test_cube)
    assert np.all(np.isfinite(test_params))

    lnpr = bt.lnprior(test_params)
    assert np.isfinite(lnpr)

    lnl = bt.lnlikelihood(test_params)
    assert np.isfinite(lnl)

    lnp = bt.lnposterior(test_params)
    assert np.isfinite(lnp) and np.isclose(lnp, lnpr + lnl)


def test_gls_exception(data_NGC6440E, data_NGC6440E_red):
    model, toas = data_NGC6440E_red
    with pytest.raises(NotImplementedError):
        bt = BayesianTiming(model, toas)


def test_badprior_exception(data_NGC6440E_default_priors):
    model, toas = data_NGC6440E_default_priors

    with pytest.raises(NotImplementedError):
        bt = BayesianTiming(model, toas)
