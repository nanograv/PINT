"""Tests for the bayesian interface."""

import io
from typing_extensions import assert_type
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
def dataJ0613m0200():
    parfile = examplefile("J0613-sim.par")
    timfile = examplefile("J0613-sim.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    set_dummy_priors(model)
    return model, toas


@pytest.fixture()
def dataJ0613m0200_efac():
    parfile = examplefile("J0613-sim.par")
    timfile = examplefile("J0613-sim.tim")
    model, toas = get_model_and_toas(parfile, timfile)

    parfile = str(model)
    parfile += "EFAC TEL gbt 1 1"
    model = get_model(io.StringIO(parfile))
    set_dummy_priors(model)

    return model, toas


def test_use_pulse_numbers(dataJ0613m0200):
    model, toas = dataJ0613m0200
    toas.compute_pulse_numbers(model)
    bt = BayesianTiming(model, toas, use_pulse_numbers=True)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl = bt.lnlikelihood(maxlike_params)
    assert not np.isnan(lnl)


def test_no_noise(dataJ0613m0200):
    model, toas = dataJ0613m0200
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl = bt.lnlikelihood(maxlike_params)
    assert bt.likelihood_method == "wls" and not np.isnan(lnl)


def test_white_noise(dataJ0613m0200_efac):
    model, toas = dataJ0613m0200_efac
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl = bt.lnlikelihood(maxlike_params)
    assert bt.likelihood_method == "wls" and not np.isnan(lnl)


def test_lnlikelihood_unit_efac(dataJ0613m0200, dataJ0613m0200_efac):
    model, toas = dataJ0613m0200
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl1 = bt.lnlikelihood(maxlike_params)

    model, toas = dataJ0613m0200_efac
    bt = BayesianTiming(model, toas)
    maxlike_params = np.array([param.value for param in bt.params], dtype=float)
    lnl2 = bt.lnlikelihood(maxlike_params)

    assert np.isclose(lnl1, lnl2)
