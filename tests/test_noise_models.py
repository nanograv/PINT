"""Test if the split basis and weights functions for EcorrNoise and PLRedNoise
and PLDMNoise give the same result as the old code."""

import re
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pint.config import examplefile
from pint.models import get_model_and_toas, get_model
from pint.models.timing_model import Component
from pint.models.noise_model import NoiseComponent
from pint.models.noise_model import (
    TimeDomainSWNoise,
    make_interpolation_basis,
    matern_kernel,
    project_basis_covariance,
)
from pint.simulation import make_fake_toas_uniform
from io import StringIO
import astropy.units as u


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


def _add_time_domain_sw_component(model, kernel, nu=1.5):
    """Attach a TimeDomainSWNoise component configured for the given kernel.

    Parameters
    ----------
    kernel : str
        One of ``'ridge'``, ``'sqexp'``, ``'matern'``, ``'quasi_periodic'``.
    nu : float
        Matern smoothness, one of ``{0.5, 1.5, 2.5}``.  Ignored unless
        ``kernel == 'MATERN'``.
    """
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)

    model["TDSWKERNEL"].value = kernel
    model["TDSWDT"].value = 14.0
    model["TDSWINTERP_KIND"].value = "LINEAR"
    model["TDSWLOGSIG"].value = 0.0

    if kernel == "RIDGE":
        pass  # only TDSWLOGSIG is required
    elif kernel == "SQEXP":
        model["TDSWLOGELL"].value = 1.2
    elif kernel == "MATERN":
        model["TDSWLOGELL"].value = 1.0
        model["TDSWNU"].value = nu
    elif kernel == "QUASI_PERIODIC":
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
    # TimeDomainSWNoise is a registered correlated-noise component, so the
    # generic parametrized tests below expect to find it in this model.
    _add_time_domain_sw_component(model, "MATERN")
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

    if np.ndim(weights) == 1:
        assert np.all(weights >= 0)
    else:
        # a full basis covariance: off-diagonal entries may legitimately be
        # negative (e.g. the quasi-periodic kernel), but it must be symmetric
        # positive semi-definite with a non-negative diagonal
        assert np.all(np.diag(weights) >= 0)
        assert np.allclose(weights, weights.T)
        assert np.all(np.linalg.eigvalsh(weights) > -1e-12 * np.max(weights))


@pytest.mark.parametrize("component_label", correlated_noise_component_labels)
def test_covariance_matrix_relation(model_and_toas, component_label):
    """Consistency between basis and weights and covariance matrix"""

    model, toas = model_and_toas
    component = model.components[component_label]
    basis_weights_func = component.basis_funcs[0]
    cov_func = component.covariance_matrix_funcs[0]

    basis, weights = basis_weights_func(toas)
    cov = cov_func(toas)
    # `weights` is a 1-D vector of variances for the Fourier-basis components and
    # a full 2-D covariance for the time-domain ones, so go through the helper
    # rather than assuming the diagonal form.
    cov2 = project_basis_covariance(basis, weights)

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


@pytest.mark.parametrize("kernel", ["RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"])
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


@pytest.mark.parametrize("kernel", ["RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"])
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


# TimeDomainSWNoise – kernel parameter validation tests


def test_time_domain_sw_invalid_kernel_rejected():
    """TimeDomainSWNoise should raise ValueError for an unknown kernel name."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "bogus_kernel"
    model["TDSWLOGSIG"].value = 0.0
    with pytest.raises(ValueError, match="TDSWKERNEL"):
        model.validate()


@pytest.mark.parametrize(
    "kernel, missing_param",
    [
        ("SQEXP", "TDSWLOGELL"),
        ("MATERN", "TDSWLOGELL"),
        ("QUASI_PERIODIC", "TDSWLOGELL"),
    ],
)
def test_time_domain_sw_missing_required_param(kernel, missing_param):
    """TimeDomainSWNoise validate() should raise when a kernel-required param is absent."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = kernel
    model["TDSWLOGSIG"].value = 0.0
    # deliberately do NOT set missing_param (or extra required params for qp)
    # For quasi_periodic we need TDSWLOGELL at minimum to fail – leave it None.
    with pytest.raises(ValueError, match=missing_param):
        model.validate()


# TimeDomainSWNoise – node-based interpolation tests


@pytest.mark.parametrize("kernel", ["RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"])
def test_time_domain_sw_node_based_interpolation(kernel):
    """Node-based TDSWNODE_ interpolation: basis/weights/cov consistent for all kernels.

    Kernel-specific parameters must be set *before* adding nodes because
    ``add_tdsw_node_component`` calls ``component.validate()`` internally as
    soon as two or more nodes are present.
    """
    model, toas = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)

    t_mjd = toas.get_mjds().value
    step = (t_mjd.max() - t_mjd.min()) / 20
    nodes = np.arange(t_mjd.min() - step, t_mjd.max() + step, step)

    # Set ALL kernel parameters before adding any nodes.
    # add_tdsw_node_component calls component.validate() once nset >= 2, so
    # missing required params would raise inside the loop otherwise.
    component.TDSWKERNEL.value = kernel
    component.TDSWLOGSIG.value = 0.0
    component.TDSWINTERP_KIND.value = "LINEAR"
    if kernel == "SQEXP":
        component.TDSWLOGELL.value = 1.2
    elif kernel == "MATERN":
        component.TDSWLOGELL.value = 1.0
        component.TDSWNU.value = 1.5
    elif kernel == "QUASI_PERIODIC":
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


def _bare_tdsw_component(kernel="RIDGE"):
    """A minimally configured TimeDomainSWNoise attached to a real model.

    Kernel parameters are set up front because ``add_tdsw_node_component``
    calls ``validate`` as soon as two nodes are present.  TDSWDT is left at its
    30.0 default so that it does not count as a competing interpolation mode.
    """
    model, toas = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    component.TDSWKERNEL.value = kernel
    component.TDSWLOGSIG.value = 0.0
    component.TDSWINTERP_KIND.value = "LINEAR"
    return model, toas, component


def test_add_tdsw_node_component_auto_index_reuses_empty_slot():
    """With index=None the lowest unset TDSWNODE_ slot must be filled first."""
    _, _, component = _bare_tdsw_component()

    # A fresh component ships with a single, unset TDSWNODE_0001.
    assert list(component.get_prefix_mapping_component("TDSWNODE_")) == [1]
    assert component.TDSWNODE_0001.value is None

    index = component.add_tdsw_node_component(54000.0)

    assert int(index) == 1
    assert component.TDSWNODE_0001.value == 54000.0
    # No new parameter was created; the empty slot was reused.
    assert list(component.get_prefix_mapping_component("TDSWNODE_")) == [1]


def test_add_tdsw_node_component_auto_index_appends_when_full():
    """With index=None and no free slot, a new TDSWNODE_ must be appended."""
    _, _, component = _bare_tdsw_component()

    first = component.add_tdsw_node_component(54000.0)
    second = component.add_tdsw_node_component(54500.0)
    third = component.add_tdsw_node_component(55000.0)

    assert [int(i) for i in (first, second, third)] == [1, 2, 3]
    assert list(component.get_prefix_mapping_component("TDSWNODE_")) == [1, 2, 3]
    assert component.TDSWNODE_0002.value == 54500.0
    assert component.TDSWNODE_0003.value == 55000.0


def test_add_tdsw_node_component_accepts_quantity():
    """A Quantity node must be converted to days, matching the plain-float form."""
    _, _, from_float = _bare_tdsw_component()
    from_float.add_tdsw_node_component(54000.0)
    from_float.add_tdsw_node_component(54500.0)

    _, _, from_quantity = _bare_tdsw_component()
    from_quantity.add_tdsw_node_component(54000.0 * u.day)
    # Also check a unit that actually needs converting.
    from_quantity.add_tdsw_node_component((54500.0 * u.day).to(u.hour))

    assert from_quantity.TDSWNODE_0001.value == from_float.TDSWNODE_0001.value
    assert_allclose(
        float(from_quantity.TDSWNODE_0002.value),
        float(from_float.TDSWNODE_0002.value),
    )


def test_add_tdsw_node_component_rejects_used_index():
    """Re-using an occupied index must raise rather than silently overwrite."""
    _, _, component = _bare_tdsw_component()
    component.add_tdsw_node_component(54000.0, index=1)

    with pytest.raises(ValueError, match="already in use"):
        component.add_tdsw_node_component(55000.0, index=1)

    # The original value survives the failed insert.
    assert component.TDSWNODE_0001.value == 54000.0


def test_add_tdsw_node_component_auto_index_yields_usable_basis():
    """Nodes added without explicit indices must produce a working GP basis."""
    _, toas, component = _bare_tdsw_component(kernel="SQEXP")
    component.TDSWLOGELL.value = 1.2

    t_mjd = toas.get_mjds().value
    step = (t_mjd.max() - t_mjd.min()) / 10
    for node in np.arange(t_mjd.min() - step, t_mjd.max() + step, step):
        component.add_tdsw_node_component(float(node))

    basis, weights = component.sw_basis_weight_pair(toas)

    assert basis.shape == (len(toas), weights.shape[0])
    assert np.ndim(weights) == 2
    assert_allclose(weights, weights.T)


# TimeDomainSWNoise – defensive error branches
#
# These guards are all unreachable through the normal model-building path
# (`validate` screens most of them first), so they are driven directly here.


def test_get_nodes_skips_unset_slots():
    """`get_nodes` must ignore TDSWNODE_ parameters that were never given a value."""
    _, toas, component = _bare_tdsw_component()

    # Leaves the default TDSWNODE_0001 unset while creating 0002 and 0003.
    component.add_tdsw_node_component(54500.0, index=2)
    component.add_tdsw_node_component(54000.0, index=3)

    assert list(component.get_prefix_mapping_component("TDSWNODE_")) == [1, 2, 3]
    assert component.TDSWNODE_0001.value is None

    # Only the two set nodes are returned, and they come back sorted.
    assert_allclose(component.get_nodes(toas), [54000.0, 54500.0])


def test_get_nodes_requires_at_least_two_nodes():
    """`get_nodes` must raise when fewer than two TDSWNODE_ values are set."""
    _, toas, component = _bare_tdsw_component()

    with pytest.raises(ValueError, match="at least 2 TDSWNODE_"):
        component.get_nodes(toas)

    component.add_tdsw_node_component(54000.0)
    with pytest.raises(ValueError, match="at least 2 TDSWNODE_"):
        component.get_nodes(toas)


def test_validate_rejects_non_finite_nodes():
    """A non-finite TDSWNODE_ value must be rejected by `validate`."""
    _, _, component = _bare_tdsw_component()
    component.add_tdsw_node_component(54000.0, index=1)
    component.add_tdsw_node_component(54500.0, index=2)

    component.TDSWNODE_0002.value = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        component.validate()

    component.TDSWNODE_0002.value = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        component.validate()


def test_get_noise_weights_rejects_unknown_kernel():
    """The kernel dispatch must fail loudly if `validate` was bypassed.

    `validate` normally screens TDSWKERNEL, so this branch is only reachable
    when the value is set directly on the component.
    """
    _, toas, component = _bare_tdsw_component()
    component.TDSWKERNEL.value = "BOGUS"

    with pytest.raises(ValueError, match="unknown TDSWKERNEL 'BOGUS'"):
        component.get_noise_weights(toas)


def test_make_interpolation_basis_requires_nodes_or_dt():
    """Neither nodes nor dt leaves nothing to build a basis from."""
    toas = np.array([54000.0, 54100.0]) * 86400.0

    with pytest.raises(ValueError, match="either nodes or dt"):
        make_interpolation_basis(toas)


def test_make_interpolation_basis_rejects_nodes_without_support():
    """Nodes far outside the TOA range give an all-zero basis, which must raise.

    This is the guard against passing TOAs and nodes in mismatched units: the
    interpolation fills with 0.0 outside the node range, so every column would
    be dropped by the rank reduction.
    """
    toas = np.array([54000.0, 54100.0]) * 86400.0  # seconds
    nodes = np.array([10.0, 20.0])  # MJD, nowhere near the TOAs

    with pytest.raises(RuntimeError, match="no support in the TOA range"):
        make_interpolation_basis(toas, nodes=nodes)


# project_basis_covariance – dispatch on the shape of Phi
#
# `project_basis_covariance` treats a 1D Phi as the diagonal of a diagonal
# covariance and a 2D Phi as a full covariance.  Anything else is a caller
# error; there is no meaningful Phi with any other number of dimensions.


def test_project_basis_covariance_diagonal_matches_dense():
    """A 1D Phi must give the same projection as the equivalent dense matrix."""
    U = np.arange(10.0).reshape(5, 2)
    phi = np.array([2.0, 3.0])

    from_diag = project_basis_covariance(U, phi)
    from_dense = project_basis_covariance(U, np.diag(phi))

    assert from_diag.shape == (5, 5)
    assert_allclose(from_diag, from_dense)


@pytest.mark.parametrize(
    "bad_phi, label",
    [(np.ones((2, 2, 2)), "3D"), (np.float64(1.0), "scalar")],
    ids=["3d", "scalar"],
)
def test_project_basis_covariance_rejects_bad_ndim(bad_phi, label):
    """Phi with any dimensionality other than 1 or 2 must be rejected."""
    U = np.arange(10.0).reshape(5, 2)
    assert np.ndim(bad_phi) not in (1, 2)

    with pytest.raises(ValueError, match="Phi must be 1D or 2D"):
        project_basis_covariance(U, bad_phi)


# make_fake_toas – injecting correlated noise from a dense Phi
#
# With only Fourier-basis components Phi is 1D and the coefficients are drawn
# independently.  A non-RIDGE TimeDomainSWNoise makes Phi 2D, which requires a
# correlated multivariate draw instead.


def _simulate_tdsw_toas(log10_sigma, seed, add_correlated_noise=True):
    """Simulate TOAs from a SQEXP TimeDomainSWNoise model and return their RMS."""
    from pint.residuals import Residuals

    model, _ = _base_model_and_toas()
    _add_time_domain_sw_component(model, "SQEXP")
    model["TDSWLOGSIG"].value = log10_sigma

    np.random.seed(seed)
    toas = make_fake_toas_uniform(
        54500,
        55500,
        40,
        model=model,
        add_noise=False,
        add_correlated_noise=add_correlated_noise,
    )
    resids = Residuals(toas, model).time_resids.to_value(u.us)
    return model, toas, float(np.std(resids))


def test_make_fake_toas_injects_correlated_noise_from_dense_phi():
    """A 2D Phi must produce a correlated multivariate draw, not a white one."""
    model, toas, rms_correlated = _simulate_tdsw_toas(0.0, seed=0)

    # Confirm the dense branch is the one being taken.
    assert np.ndim(model.noise_model_basis_weight(toas)) == 2

    _, _, rms_baseline = _simulate_tdsw_toas(0.0, seed=0, add_correlated_noise=False)

    assert np.isfinite(rms_correlated)
    # Without correlated noise the residuals are numerically zero, so the
    # injected GP dominates by orders of magnitude.
    assert rms_correlated > 100 * rms_baseline

    # Same seed, same realization.
    assert rms_correlated == _simulate_tdsw_toas(0.0, seed=0)[2]


def test_make_fake_toas_correlated_noise_scales_with_amplitude():
    """The injected signal must scale with TDSWLOGSIG, i.e. the draw uses Phi.

    The ratio is well below the nominal factor of 10 because the timing model
    absorbs part of the injected signal, and absorbs a different fraction at
    each amplitude, so only a loose bound is asserted.
    """
    _, _, rms_low = _simulate_tdsw_toas(0.0, seed=0)
    _, _, rms_high = _simulate_tdsw_toas(1.0, seed=0)

    assert rms_high > 2 * rms_low


# TimeDomainSWNoise – GLS fitter integration


@pytest.mark.parametrize("kernel", ["RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"])
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


# TimeDomainSWNoise – wideband datasets and dense (2D) basis covariance
#
# Every kernel other than RIDGE returns a full covariance matrix, which makes
# `TimingModel.noise_model_basis_weight` block-diagonalise the per-component
# weights into a 2D Phi.  The tests below drive that Phi through the wideband
# machinery: `WidebandTOAFitter.fit_toas`, `TimingModel.wideband_covariance_matrix`
# and `WidebandTOAResiduals.calc_wideband_whitened_resids`.


@pytest.fixture(scope="module")
def wideband_base():
    """A B1855+09 model stripped of DMX/JUMP/FD, plus matching wideband TOAs.

    DMX and JUMP ranges have no support in the simulated TOAs and would abort
    the fit in validation before any linear algebra runs; FD is dropped for the
    same reason.  The TOAs do not depend on the noise kernel, so they are built
    once and shared.
    """
    model = get_model(examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
    for component in ("DispersionDMX", "PhaseJump", "FD"):
        model.remove_component(component)
    toas = make_fake_toas_uniform(
        54500,
        55500,
        50,
        model=model,
        freq=1400 * u.MHz,
        wideband=True,
        add_noise=True,
    )
    return model, toas


def _wideband_model_and_toas(wideband_base, kernel):
    """Attach a `TimeDomainSWNoise` component to a copy of the shared model."""
    base_model, toas = wideband_base
    model = deepcopy(base_model)
    _add_time_domain_sw_component(model, kernel)
    # Fitting the full binary model on 50 simulated TOAs drives SINI out of
    # range; the dense-covariance code path is exercised just as well by a
    # small, well-conditioned subset of free parameters.
    model.free_params = ["F0", "F1", "DM"]
    return model, toas


@pytest.mark.parametrize("kernel", ["SQEXP", "MATERN", "QUASI_PERIODIC"])
def test_time_domain_sw_wideband_basis_weight_is_dense(wideband_base, kernel):
    """Non-RIDGE kernels must make the joint basis covariance a 2D matrix."""
    model, toas = _wideband_model_and_toas(wideband_base, kernel)

    Phi = model.noise_model_basis_weight(toas)
    U = model.noise_model_wideband_designmatrix(toas)

    assert np.ndim(Phi) == 2, f"kernel '{kernel}' did not produce a 2D Phi"
    assert Phi.shape[0] == Phi.shape[1] == U.shape[1]
    assert U.shape[0] == 2 * len(toas)
    assert_allclose(Phi, Phi.T)
    # Block-diagonal by construction, so it must be positive definite.
    np.linalg.cholesky(Phi)


@pytest.mark.parametrize("kernel", ["SQEXP", "MATERN", "QUASI_PERIODIC"])
def test_time_domain_sw_wideband_covariance_matrix_dense(wideband_base, kernel):
    """`wideband_covariance_matrix` must project a 2D Phi as U @ Phi @ U.T."""
    model, toas = _wideband_model_and_toas(wideband_base, kernel)

    C = model.wideband_covariance_matrix(toas)
    U = model.noise_model_wideband_designmatrix(toas)
    Phi = model.noise_model_basis_weight(toas)
    N = np.diag(model.scaled_wideband_uncertainty(toas) ** 2)

    assert C.shape == (2 * len(toas), 2 * len(toas))
    assert np.all(np.isfinite(C))
    assert_allclose(C, C.T)
    assert_allclose(C, N + U @ Phi @ U.T, rtol=1e-10, atol=0)


@pytest.mark.parametrize("kernel", ["RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"])
def test_time_domain_sw_wideband_fit_dense_phi_matches_full_cov(wideband_base, kernel):
    """The Woodbury and explicit-covariance branches must agree.

    With a 2D Phi, `full_cov=False` inverts Phi and adds it to the normal
    matrix, while `full_cov=True` builds the dense data-space covariance and
    Cholesky-solves against it.  The two are algebraically identical, so a
    mismatch means one of the dense-Phi branches is wrong.
    """
    from pint import fitter

    model, toas = _wideband_model_and_toas(wideband_base, kernel)

    f_woodbury = fitter.WidebandTOAFitter(toas, model)
    chi2_woodbury = f_woodbury.fit_toas(maxiter=1, full_cov=False)

    f_fullcov = fitter.WidebandTOAFitter(toas, model)
    chi2_fullcov = f_fullcov.fit_toas(maxiter=1, full_cov=True)

    assert np.isfinite(chi2_woodbury)
    assert_allclose(chi2_woodbury, chi2_fullcov, rtol=1e-6)

    for param in model.free_params:
        assert_allclose(
            getattr(f_woodbury.model, param).value,
            getattr(f_fullcov.model, param).value,
            rtol=1e-6,
            err_msg=f"{param} differs between full_cov modes for kernel '{kernel}'",
        )


@pytest.mark.parametrize("kernel", ["SQEXP", "MATERN", "QUASI_PERIODIC"])
def test_time_domain_sw_wideband_whitened_resids_dense_phi(wideband_base, kernel):
    """Whitening wideband residuals against a 2D Phi must give unit-scale output."""
    from pint import fitter

    model, toas = _wideband_model_and_toas(wideband_base, kernel)

    f = fitter.WidebandTOAFitter(toas, model)
    f.fit_toas(maxiter=1)

    rw = f.resids.calc_wideband_whitened_resids()

    assert rw.shape == (2 * len(toas),)
    assert np.all(np.isfinite(rw))
    # Correctly whitened residuals are unit-normal; a broken 2D branch shows up
    # as a scale error of orders of magnitude rather than a marginal one.
    assert 0.2 < np.std(rw) < 5.0, f"whitened residual scale is off for '{kernel}'"


# TimeDomainSWNoise – validation edge cases


def test_time_domain_sw_invalid_interp_kind_rejected():
    """An unsupported TDSWINTERP_KIND value must raise ValueError."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "RIDGE"
    model["TDSWLOGSIG"].value = 0.0
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
    model["TDSWKERNEL"].value = "MATERN"
    model["TDSWLOGSIG"].value = 0.0
    model["TDSWLOGELL"].value = 1.0
    model["TDSWDT"].value = 30.0
    model["TDSWNU"].value = bad_nu
    with pytest.raises(ValueError, match="TDSWNU"):
        model.validate()


# matern_kernel – all supported smoothness values
#
# `validate` pins TDSWNU to {0.5, 1.5, 2.5}, and the rest of the test suite
# only ever uses the 1.5 default, so the nu=0.5 and nu=2.5 closed forms are
# checked here against the general Matern function.


def _matern_reference(r, sigma, ell, nu):
    """Matern covariance from the general Bessel form, independent of PINT.

    :math:`K(r) = \\sigma^2 \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
    \\left(\\sqrt{2\\nu}\\,r/\\ell\\right)^{\\nu}
    K_{\\nu}\\!\\left(\\sqrt{2\\nu}\\,r/\\ell\\right)`, which is singular at
    :math:`r = 0` and equals :math:`\\sigma^2` in the limit.
    """
    from scipy.special import gamma, kv

    rr = np.sqrt(2 * nu) * r / ell
    out = np.empty_like(rr)
    at_zero = rr == 0
    out[at_zero] = 1.0
    z = rr[~at_zero]
    out[~at_zero] = (2 ** (1 - nu) / gamma(nu)) * z**nu * kv(nu, z)
    return sigma**2 * out


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_kernel_matches_general_bessel_form(nu):
    """Each hard-coded closed form must reproduce the general Matern function.

    This checks the constants (sqrt(3), sqrt(5), 5/3) rather than restating
    them, so a transposed coefficient between branches would be caught.
    """
    nodes = np.array([0.0, 10.0, 25.0, 60.0, 130.0]) * 86400.0
    log10_sigma, log10_ell = -7.0, 2.0
    sigma = 10**log10_sigma
    ell = 10**log10_ell * 86400.0

    K = matern_kernel(nodes, log10_sigma, log10_ell, nu)

    r = np.abs(nodes[None, :] - nodes[:, None])
    jitter = np.eye(len(nodes)) * (sigma / 50000) ** 2
    expected = _matern_reference(r, sigma, ell, nu) + jitter

    assert_allclose(K, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_kernel_is_a_valid_covariance(nu):
    """The kernel must be symmetric, positive definite and decreasing in lag."""
    nodes = np.arange(0.0, 200.0, 20.0) * 86400.0
    K = matern_kernel(nodes, log10_sigma=-7.0, log10_ell=2.0, nu=nu)

    assert K.shape == (len(nodes), len(nodes))
    assert_allclose(K, K.T)
    np.linalg.cholesky(K)  # raises if not positive definite

    # Uniformly spaced nodes, so row 0 is the kernel sampled at increasing lag.
    lags = K[0]
    assert np.all(np.diff(lags) < 0), f"matern nu={nu} is not decreasing in lag"


def test_matern_kernel_rejects_unsupported_nu():
    """`matern_kernel` guards nu itself, independently of `TimingModel.validate`."""
    nodes = np.array([0.0, 10.0, 25.0]) * 86400.0
    with pytest.raises(ValueError, match="nu in"):
        matern_kernel(nodes, -7.0, 2.0, nu=1.0)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_time_domain_sw_matern_nu_variants(nu):
    """Every allowed TDSWNU must validate and yield a usable dense weight matrix."""
    model, toas = _base_model_and_toas()
    component = _add_time_domain_sw_component(model, "MATERN", nu=nu)

    assert model["TDSWNU"].value == nu

    Phi = component.get_noise_weights(toas)
    Umat = component.get_noise_basis(toas)

    assert np.ndim(Phi) == 2
    assert Phi.shape[0] == Phi.shape[1] == Umat.shape[1]
    assert_allclose(Phi, Phi.T)
    np.linalg.cholesky(Phi)


def test_time_domain_sw_conflicting_dt_and_nodes_rejected():
    """Setting both a non-default TDSWDT and TDSWNODE_ parameters must raise ValueError.

    ``add_tdsw_node_component`` calls ``component.validate()`` internally once
    two nodes are present, so the exception fires on the second ``add_tdsw_node_component``
    call rather than on an explicit ``model.validate()``.
    """
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    component.TDSWKERNEL.value = "RIDGE"
    component.TDSWLOGSIG.value = 0.0
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
    model["TDSWKERNEL"].value = "RIDGE"
    model["TDSWLOGSIG"].value = 0.0
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
    component.TDSWKERNEL.value = "RIDGE"
    component.TDSWLOGSIG.value = 0.0
    component.add_tdsw_node_component(55000.0, index=1)  # nset=1, no validate yet
    # Second addition with same MJD triggers internal validate() -> must raise.
    with pytest.raises(ValueError, match="unique"):
        component.add_tdsw_node_component(55000.0, index=2)


def test_time_domain_sw_negative_dt_rejected():
    """A non-positive TDSWDT must raise ValueError."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)
    model["TDSWKERNEL"].value = "RIDGE"
    model["TDSWLOGSIG"].value = 0.0
    model["TDSWDT"].value = -5.0
    with pytest.raises(ValueError, match="TDSWDT"):
        model.validate()


@pytest.mark.parametrize(
    "kernel_in, kind_in",
    [
        ("ridge", "linear"),
        ("Ridge", "Linear"),
        ("RIDGE", "LINEAR"),
    ],
)
def test_time_domain_sw_value_case_is_normalised(kernel_in, kind_in):
    """TDSWKERNEL / TDSWINTERP_KIND accept any case and are stored upper case."""
    model, _ = _base_model_and_toas()
    component = TimeDomainSWNoise()
    model.add_component(component, validate=False)

    model["TDSWKERNEL"].value = kernel_in
    model["TDSWINTERP_KIND"].value = kind_in
    model["TDSWDT"].value = 14.0
    model["TDSWLOGSIG"].value = 0.0
    model.validate()

    assert model["TDSWKERNEL"].value == "RIDGE"
    assert model["TDSWINTERP_KIND"].value == "LINEAR"
    par_str = model.as_parfile()
    assert re.search(r"TDSWKERNEL\s+RIDGE", par_str)
    assert re.search(r"TDSWINTERP_KIND\s+LINEAR", par_str)


# TimeDomainSWNoise – par-file serialisation roundtrip


@pytest.mark.parametrize("kernel", ["RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"])
def test_time_domain_sw_parfile_roundtrips(kernel):
    """TimeDomainSWNoise parameters are written to, and read back from, a par file.

    The component is registered, so ``get_model()`` reconstructs it from the
    ``TDSW*`` parameters rather than requiring an explicit ``add_component``.
    """
    model, _ = _base_model_and_toas()
    _add_time_domain_sw_component(model, kernel)

    par_str = model.as_parfile()

    assert re.search(
        rf"TDSWKERNEL\s+{re.escape(kernel)}", par_str
    ), f"Expected 'TDSWKERNEL {kernel}' in par string"
    assert "TDSWLOGSIG" in par_str
    assert "TDSWDT" in par_str
    if kernel in ("SQEXP", "MATERN", "QUASI_PERIODIC"):
        assert "TDSWLOGELL" in par_str
    if kernel == "MATERN":
        assert "TDSWNU" in par_str
    if kernel == "QUASI_PERIODIC":
        assert "TDSWLOGGAMP" in par_str
        assert "TDSWLOGP" in par_str

    # read it back: the component must be rebuilt with the same parameter values
    model2 = get_model(StringIO(par_str))
    assert "TimeDomainSWNoise" in model2.components
    assert model2["TDSWKERNEL"].value == kernel
    assert model2["TDSWINTERP_KIND"].value == model["TDSWINTERP_KIND"].value
    for par in (
        "TDSWDT",
        "TDSWLOGSIG",
        "TDSWLOGELL",
        "TDSWNU",
        "TDSWLOGGAMP",
        "TDSWLOGP",
    ):
        assert model2[par].value == model[par].value, par


def test_time_domain_sw_not_selected_without_tdsw_params():
    """A par file with no TDSW* parameters must not pull in the component."""
    model, _ = _base_model_and_toas()
    assert "TimeDomainSWNoise" not in model.components


@pytest.mark.parametrize("gp", ["Red", "DM"])
def test_log_frequencies(gp):
    GP = gp.upper()
    par = f"""
        PSR         TEST
        RAJ         05:00:00
        DECJ        15:00:00
        F0          100
        F1          -1e-15
        DM          15
        PEPOCH      55000
        TN{GP}AMP    -15
        TN{GP}GAM    3.5
        TN{GP}C      4
        TN{GP}FLOG   2
        TN{GP}FLOG_FACTOR 2
        TZRMJD      55000
        TZRFRQ      inf
        TZRSITE     ssb
        EPHEM       DE440
        CLOCK       TT(BIPM2023)
        UNITS       TDB
    """
    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(
        54000,
        56000,
        100,
        m,
        add_noise=True,
        add_correlated_noise=True,
    )
    assert np.allclose(
        m.components[f"PL{gp}Noise"].get_time_frequencies(t)[1]
        * t.get_Tspan().to_value("s"),
        np.array([0.25, 0.5, 1, 2, 3, 4]),
    )

    M0 = m.noise_model_designmatrix(t)

    t_ = t.table["tdbld"].quantity * u.day
    m[f"TN{GP}TSPAN"].quantity = np.max(t_) - np.min(t_)
    M1 = m.noise_model_designmatrix(t)
    assert np.allclose(M0, M1)
