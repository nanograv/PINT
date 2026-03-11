import numpy as np

from pint.models.noise_model import (
    create_fourier_design_matrix,
    get_rednoise_freqs,
    matern_kernel,
    periodic_kernel,
    powerlaw,
    ridge_kernel,
    se_kernel,
)


def test_rednoise_freqs_from_longdouble_times_are_float64():
    t = np.array([0.0, 10.0, 20.0], dtype=np.longdouble)
    freqs = get_rednoise_freqs(t, nmodes=4, Tspan=np.max(t) - np.min(t))
    assert np.asarray(freqs).dtype == np.float64


def test_powerlaw_from_longdouble_freqs_is_float64():
    freqs = np.array([0.1, 0.2, 0.3], dtype=np.longdouble)
    weights = powerlaw(freqs, A=1e-15, gamma=4.0)
    assert np.asarray(weights).dtype == np.float64


def test_fourier_design_matrix_dtype_remains_float64():
    t = np.array([1.0, 2.0, 3.0], dtype=np.longdouble)
    freqs = np.array([0.1, 0.2], dtype=np.longdouble)
    design = create_fourier_design_matrix(t, freqs)
    assert design.dtype == np.float64


def test_time_domain_kernel_weights_from_longdouble_nodes_are_float64():
    nodes = np.array([1.0, 2.0, 4.0], dtype=np.longdouble)

    assert ridge_kernel(nodes).dtype == np.float64
    assert se_kernel(nodes).dtype == np.float64
    assert matern_kernel(nodes).dtype == np.float64
    assert periodic_kernel(nodes).dtype == np.float64
