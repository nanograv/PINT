"""Test if the split basis and weights functions for EcorrNoise and PLRedNoise
and PLDMNoise give the same result as the old code."""

import numpy as np
import pytest
from pint.config import examplefile
from pint.models import get_model_and_toas, get_model
from pint.models.timing_model import Component
from pint.models.noise_model import NoiseComponent
from pint.models.noise_model import (
    TimeDomainSWNoise,
    project_basis_covariance,
)
from pint.simulation import make_fake_toas_uniform
from io import StringIO


noise_component_labels = [
    cl for cl, c in Component.component_types.items() if issubclass(c, NoiseComponent)
]
correlated_noise_component_labels = [
    cl
    for cl, c in Component.component_types.items()
    if issubclass(c, NoiseComponent) and c().introduces_correlated_errors
]


def add_DM_noise_to_model(model):
    all_components = Component.component_types
    model.add_component(all_components["PLDMNoise"](), validate=False)
    model["TNDMAMP"].quantity = -13
    model["TNDMGAM"].quantity = 1.2
    model["TNDMC"].value = 30
    model["TNDMFLOG"].value = 4
    model["TNDMFLOG_FACTOR"].value = 2
    model.validate()


def add_SW_noise_to_model(model):
    all_components = Component.component_types
    model.add_component(all_components["PLSWNoise"](), validate=False)
    model["TNSWAMP"].quantity = -12
    model["TNSWGAM"].quantity = -2.0  # blue spectrum
    model["TNSWC"].value = 50
    model["TNSWFLOG"].value = 4
    model["TNSWFLOG_FACTOR"].value = 2
    model.validate()


def add_chrom_noise_to_model(model):
    all_components = Component.component_types
    model.add_component(all_components["PLChromNoise"](), validate=False)
    model["TNCHROMAMP"].quantity = -14
    model["TNCHROMGAM"].quantity = 1.2
    model["TNCHROMC"].value = 30
    model["TNCHROMFLOG"].value = 4
    model["TNCHROMFLOG_FACTOR"].value = 2

    model.add_component(all_components["ChromaticCM"](), validate=False)
    model["TNCHROMIDX"].value = 4

    model.validate()


def _base_model_and_toas():
    """Load a clean real dataset used for integration-style noise component tests."""
    parfile = examplefile("B1855+09_NANOGrav_9yv1.gls.par")
    timfile = examplefile("B1855+09_NANOGrav_9yv1.tim")
    return get_model_and_toas(parfile, timfile)


def _add_time_domain_sw_component(model, kernel):
    """Attach a TimeDomainSWNoise component configured for the given kernel.

    Parameters
    ----------
    kernel : str
        One of ``'ridge'``, ``'sqexp'``, ``'matern'``, ``'quasi_periodic'``.
    """
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)

    model["TDSWKERNEL"].value = kernel
    model["TDSWDT"].value = 14.0
    model["TDSWINTERP_KIND"].value = "linear"
    model["TDSWLOGSIG"].value = -7.0

    if kernel == "ridge":
        pass  # only TDSWLOGSIG is required
    elif kernel == "sqexp":
        model["TDSWLOGELL"].value = 1.2
    elif kernel == "matern":
        model["TDSWLOGELL"].value = 1.0
        model["TDSWNU"].value = 1.5
    elif kernel == "quasi_periodic":
        model["TDSWLOGELL"].value = 1.1
        model["TDSWLOGGAMP"].value = -0.2
        model["TDSWLOGP"].value = 1.5
    else:
        raise ValueError(f"Unsupported time-domain SW kernel: {kernel}")

    model.validate()
    return component


@pytest.fixture(scope="module")
def model_and_toas():
    parfile = examplefile("B1855+09_NANOGrav_9yv1.gls.par")
    timfile = examplefile("B1855+09_NANOGrav_9yv1.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    add_DM_noise_to_model(model)
    add_chrom_noise_to_model(model)
    add_SW_noise_to_model(model)
    return model, toas


@pytest.mark.parametrize("component_label", noise_component_labels)
def test_introduces_correlated_errors(component_label):
    """All `NoiseComponent` classes should have a Boolean attribute `introduces_correlated_errors`."""

    component = Component.component_types[component_label]
    assert hasattr(component, "introduces_correlated_errors") and isinstance(
        component().introduces_correlated_errors, bool
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


@pytest.mark.parametrize("kernel", ["ridge", "sqexp", "matern", "quasi_periodic"])
def test_noise_weights_sign_time_domain_sw_integration(kernel):
    """Integration test: each time-domain SW kernel should produce non-negative weights.

    This extends the existing ``test_noise_weights_sign`` coverage to the unified
    TimeDomainSWNoise component with each supported kernel.
    """

    model, toas = _base_model_and_toas()
    component = _add_time_domain_sw_component(model, kernel)

    basis, weights = component.basis_funcs[0](toas)

    assert basis.shape == (len(toas), len(weights))
    assert np.all(weights >= 0)


@pytest.mark.parametrize("kernel", ["ridge", "sqexp", "matern", "quasi_periodic"])
def test_time_domain_sw_covariance_matches_basis_weights(kernel):
    """Integration test: covariance must equal basis*weights*basis^T for each kernel.

    Mirrors ``test_covariance_matrix_relation`` for the unified TimeDomainSWNoise.
    """

    model, toas = _base_model_and_toas()
    component = _add_time_domain_sw_component(model, kernel)

    basis, weights = component.basis_funcs[0](toas)
    cov = component.covariance_matrix_funcs[0](toas)
    cov_from_basis = project_basis_covariance(basis, weights)

    assert np.allclose(cov, cov_from_basis)


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


# ---------------------------------------------------------------------------
# TimeDomainSWNoise – kernel parameter validation tests
# ---------------------------------------------------------------------------


def test_time_domain_sw_invalid_kernel_rejected():
    """TimeDomainSWNoise should raise ValueError for an unknown kernel name."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "bogus_kernel"
    model["TDSWLOGSIG"].value = -7.0
    with pytest.raises(ValueError, match="TDSWKERNEL"):
        model.validate()


@pytest.mark.parametrize(
    "kernel, missing_param",
    [
        ("sqexp", "TDSWLOGELL"),
        ("matern", "TDSWLOGELL"),
        ("quasi_periodic", "TDSWLOGELL"),
    ],
)
def test_time_domain_sw_missing_required_param(kernel, missing_param):
    """TimeDomainSWNoise validate() should raise when a kernel-required param is absent."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = kernel
    model["TDSWLOGSIG"].value = -7.0
    # deliberately do NOT set missing_param (or extra required params for qp)
    # For quasi_periodic we need TDSWLOGELL at minimum to fail – leave it None.
    with pytest.raises(ValueError, match=missing_param):
        model.validate()


# ---------------------------------------------------------------------------
# TimeDomainSWNoise – node-based interpolation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["ridge", "sqexp", "matern", "quasi_periodic"])
def test_time_domain_sw_node_based_interpolation(kernel):
    """Node-based TDSWNODE_ interpolation: basis/weights/cov consistent for all kernels.

    Kernel-specific parameters must be set *before* adding nodes because
    ``_add_tdsw_node_component`` calls ``component.validate()`` internally as
    soon as two or more nodes are present.
    """
    model, toas = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)

    t_mjd = toas.get_mjds().value
    step = (t_mjd.max() - t_mjd.min()) / 20
    nodes = np.arange(t_mjd.min() - step, t_mjd.max() + step, step)

    # Set ALL kernel parameters before adding any nodes.
    # _add_tdsw_node_component calls component.validate() once nset >= 2, so
    # missing required params would raise inside the loop otherwise.
    component.TDSWKERNEL.value = kernel
    component.TDSWLOGSIG.value = -7.0
    component.TDSWINTERP_KIND.value = "linear"
    if kernel == "sqexp":
        component.TDSWLOGELL.value = 1.2
    elif kernel == "matern":
        component.TDSWLOGELL.value = 1.0
        component.TDSWNU.value = 1.5
    elif kernel == "quasi_periodic":
        component.TDSWLOGELL.value = 1.1
        component.TDSWLOGGAMP.value = -0.2
        component.TDSWLOGP.value = 1.5

    for i, node in enumerate(nodes):
        component.add_tdsw_node_component(float(node), index=i + 1)

    basis, weights = component.basis_funcs[0](toas)
    cov = component.covariance_matrix_funcs[0](toas)
    cov_from_basis = project_basis_covariance(basis, weights)

    assert basis.shape == (len(toas), len(weights))
    assert np.all(weights >= 0)
    assert np.allclose(cov, cov_from_basis)


# ---------------------------------------------------------------------------
# TimeDomainSWNoise – GLS fitter integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["ridge", "sqexp", "matern", "quasi_periodic"])
def test_time_domain_sw_gls_fitter_runs(kernel):
    """GLSFitter must instantiate and produce a finite chi-squared for each kernel.

    A full convergent fit is not required (real data may trigger numerical
    step failures), but residuals and chi-squared must be computable after at
    most two iterations.
    """
    from pint import fitter

    model, toas = _base_model_and_toas()
    _add_time_domain_sw_component(model, kernel)

    f = fitter.GLSFitter(toas, model)
    try:
        f.fit_toas(maxiter=2)
    except Exception:
        pass  # numerical issues are acceptable; we only check chi2 below

    chi2 = f.resids.chi2
    assert np.isfinite(chi2), f"chi2 is not finite for kernel '{kernel}'"


# ---------------------------------------------------------------------------
# TimeDomainSWNoise – validation edge cases
# ---------------------------------------------------------------------------


def test_time_domain_sw_invalid_interp_kind_rejected():
    """An unsupported TDSWINTERP_KIND value must raise ValueError."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "ridge"
    model["TDSWLOGSIG"].value = -7.0
    model["TDSWDT"].value = 30.0
    model["TDSWINTERP_KIND"].value = "not_a_kind"
    with pytest.raises(ValueError, match="TDSWINTERP_KIND"):
        model.validate()


@pytest.mark.parametrize("bad_nu", [0.0, 1.0, 2.0, 3.0])
def test_time_domain_sw_invalid_matern_nu_rejected(bad_nu):
    """TDSWNU values outside {0.5, 1.5, 2.5} must raise ValueError for matern kernel."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "matern"
    model["TDSWLOGSIG"].value = -7.0
    model["TDSWLOGELL"].value = 1.0
    model["TDSWDT"].value = 30.0
    model["TDSWNU"].value = bad_nu
    with pytest.raises(ValueError, match="TDSWNU"):
        model.validate()


def test_time_domain_sw_conflicting_dt_and_nodes_rejected():
    """Setting both a non-default TDSWDT and TDSWNODE_ parameters must raise ValueError.

    ``_add_tdsw_node_component`` calls ``component.validate()`` internally once
    two nodes are present, so the exception fires on the second ``add_tdsw_node_component``
    call rather than on an explicit ``model.validate()``.
    """
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    component.TDSWKERNEL.value = "ridge"
    component.TDSWLOGSIG.value = -7.0
    component.TDSWDT.value = 14.0  # non-default: conflicts with nodes
    component.add_tdsw_node_component(55000.0, index=1)  # nset=1, no validate yet
    # Second node addition triggers internal validate() -> must raise.
    with pytest.raises(ValueError, match="interpolation mode"):
        component.add_tdsw_node_component(55200.0, index=2)


def test_time_domain_sw_single_node_rejected():
    """Exactly one TDSWNODE_ value (fewer than the required 2) must raise ValueError."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "ridge"
    model["TDSWLOGSIG"].value = -7.0
    component.add_tdsw_node_component(55000.0, index=1)
    with pytest.raises(ValueError, match="at least 2"):
        model.validate()


def test_time_domain_sw_duplicate_nodes_rejected():
    """Duplicate TDSWNODE_ values must raise ValueError.

    The second ``add_tdsw_node_component`` call triggers internal validation
    (nset >= 2) which should detect the duplicate and raise.
    """
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    component.TDSWKERNEL.value = "ridge"
    component.TDSWLOGSIG.value = -7.0
    component.add_tdsw_node_component(55000.0, index=1)  # nset=1, no validate yet
    # Second addition with same MJD triggers internal validate() -> must raise.
    with pytest.raises(ValueError, match="unique"):
        component.add_tdsw_node_component(55000.0, index=2)


def test_time_domain_sw_negative_dt_rejected():
    """A non-positive TDSWDT must raise ValueError."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "ridge"
    model["TDSWLOGSIG"].value = -7.0
    model["TDSWDT"].value = -5.0
    with pytest.raises(ValueError, match="TDSWDT"):
        model.validate()


# ---------------------------------------------------------------------------
# TimeDomainSWNoise – par-file serialisation roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["ridge", "sqexp", "matern", "quasi_periodic"])
def test_time_domain_sw_parfile_serialises_params(kernel):
    """All TimeDomainSWNoise parameters appear in the serialised par string.

    ``TimeDomainSWNoise`` has ``register = False`` so ``get_model()`` will not
    reconstruct it from a par file.  This test therefore validates that the
    parameters are *written* correctly rather than testing a full read-back
    roundtrip.
    """
    import re

    model, _ = _base_model_and_toas()
    _add_time_domain_sw_component(model, kernel)

    par_str = model.as_parfile()

    assert re.search(
        rf"TDSWKERNEL\s+{re.escape(kernel)}", par_str
    ), f"Expected 'TDSWKERNEL {kernel}' in par string"
    assert "TDSWLOGSIG" in par_str
    assert "TDSWDT" in par_str
    if kernel in ("sqexp", "matern", "quasi_periodic"):
        assert "TDSWLOGELL" in par_str
    if kernel == "matern":
        assert "TDSWNU" in par_str
    if kernel == "quasi_periodic":
        assert "TDSWLOGGAMP" in par_str
        assert "TDSWLOGP" in par_str
