"""Test if the split basis and weights functions for EcorrNoise and PLRedNoise
and PLDMNoise give the same result as the old code."""

import numpy as np
import pytest
from pint.config import examplefile
from pint.models import get_model_and_toas, get_model
from pint.models.timing_model import Component
from pint.models.noise_model import NoiseComponent
from pint.simulation import make_fake_toas_uniform
from io import StringIO


noise_component_labels = [
    cl for cl, c in Component.component_types.items() if issubclass(c, NoiseComponent)
]
correlated_noise_component_labels = [
    cl
    for cl, c in Component.component_types.items()
    if issubclass(c, NoiseComponent)
    and hasattr(c, "introduces_correlated_errors")
    and c.introduces_correlated_errors
]


def add_DM_noise_to_model(model):
    all_components = Component.component_types
    noise_class = all_components["PLDMNoise"]
    noise = noise_class()  # Make the DM noise instance.
    model.add_component(noise, validate=False)
    model["TNDMAMP"].quantity = 1e-13
    model["TNDMGAM"].quantity = 1.2
    model["TNDMC"].value = 30
    model.validate()
    return model


@pytest.fixture()
def model_and_toas():
    parfile = examplefile("B1855+09_NANOGrav_9yv1.gls.par")
    timfile = examplefile("B1855+09_NANOGrav_9yv1.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    model = add_DM_noise_to_model(model)
    return model, toas


@pytest.mark.parametrize("component_label", noise_component_labels)
def test_introduces_correlated_errors(component_label):
    """All `NoiseComponent` classes should have a Boolean attribute `introduces_correlated_errors`."""

    component = Component.component_types[component_label]
    assert hasattr(component, "introduces_correlated_errors") and isinstance(
        component.introduces_correlated_errors, bool
    )


@pytest.mark.parametrize("component_label", correlated_noise_component_labels)
def test_noise_basis_shape(model_and_toas, component_label):
    """Test shape of basis matrix."""

    model, toas = model_and_toas
    component = model.components[component_label]
    basis_weight_func = component.basis_funcs[0]

    basis, weights = basis_weight_func(toas)

    assert basis.shape == (len(toas), len(weights))


@pytest.mark.parametrize("component_label", correlated_noise_component_labels)
def test_noise_weights_sign(model_and_toas, component_label):
    """Weights should be positive."""

    model, toas = model_and_toas
    component = model.components[component_label]
    basis_weight_func = component.basis_funcs[0]

    basis, weights = basis_weight_func(toas)

    assert np.all(weights >= 0)


@pytest.mark.parametrize("component_label", correlated_noise_component_labels)
def test_covariance_matrix_relation(model_and_toas, component_label):
    """Consistency between basis and weights and covariance matrix"""

    model, toas = model_and_toas
    component = model.components[component_label]
    basis_weights_func = component.basis_funcs[0]
    cov_func = component.covariance_matrix_funcs[0]

    basis, weights = basis_weights_func(toas)
    cov = cov_func(toas)
    cov2 = np.dot(basis * weights[None, :], basis.T)

    assert np.allclose(cov, cov2)


def test_ecorrnoise_basis_integer(model_and_toas):
    """ECORR basis matrix contains positive integers."""

    model, toas = model_and_toas
    ecorrcomponent = model.components["EcorrNoise"]

    basis, weights = ecorrcomponent.ecorr_basis_weight_pair(toas)

    assert np.all(basis.astype(int) == basis) and np.all(basis >= 0)


@pytest.mark.parametrize("component_label", correlated_noise_component_labels)
def test_noise_basis_weights_funcs(model_and_toas, component_label):
    model, toas = model_and_toas

    component = model.components[component_label]

    basis_weights_func = component.basis_funcs[0]

    basis, weights = basis_weights_func(toas)

    basis_ = component.get_noise_basis(toas)
    weights_ = component.get_noise_weights(toas)

    assert np.allclose(basis_, basis) and np.allclose(weights, weights_)


def test_white_noise_model_derivs():
    par = """
        ELAT    1.3     1
        ELONG   2.5     1
        F0      100     1
        F1      1e-13   1
        PEPOCH  55000
        EFAC mjd 49999 53000 2      1
        EQUAD mjd 49999 53000 0.5      1
        EFAC mjd 53000 55000 1.5    1
        EQUAD mjd 53000 55000 1    1
    """
    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(50000, 55000, 1000, m, add_noise=True)

    dq1 = m.d_toasigma_d_param(t, "EQUAD1")
    dq2 = m.d_toasigma_d_param(t, "EQUAD2")
    df1 = m.d_toasigma_d_param(t, "EFAC1")
    df2 = m.d_toasigma_d_param(t, "EFAC2")

    sigma = m.scaled_toa_uncertainty(t)

    assert np.isclose(df1[0], sigma[0] / m.EFAC1.quantity)
    assert np.isclose(df2[-1], sigma[-1] / m.EFAC2.quantity)

    assert np.isclose(
        dq1[0], sigma[0] * m.EQUAD1.quantity * (m.EFAC1.quantity / sigma[0]) ** 2
    )
    assert np.isclose(
        dq2[-1], sigma[-1] * m.EQUAD2.quantity * (m.EFAC2.quantity / sigma[-1]) ** 2
    )
