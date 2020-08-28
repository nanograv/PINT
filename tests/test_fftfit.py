from functools import wraps

import numpy as np
import pytest
import scipy.stats
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    complex_numbers,
    composite,
    floats,
    fractions,
    integers,
    just,
    one_of,
)
from numpy.testing import assert_allclose, assert_array_almost_equal

import pint.profile.fftfit_aarchiba as fftfit
from pint.profile import fftfit_aarchiba
from pint.profile import fftfit_nustar


def assert_rms_close(a, b, rtol=1e-8, atol=1e-8):
    assert np.mean((a - b) ** 2) < rtol * (np.mean(a ** 2) + np.mean(b ** 2)) + atol


def assert_allclose_phase(a, b, atol=1e-8):
    assert np.all(np.abs(fftfit.wrap(a - b)) <= atol)


@composite
def powers_of_two(draw):
    return 2 ** draw(integers(4, 16))


@composite
def vonmises_templates(draw, ns=powers_of_two(), phase=floats(0, 1)):
    return fftfit.vonmises_profile(draw(floats(1, 1000)), draw(ns), draw(phase))


@composite
def vonmises_templates_noisy(draw, ns=powers_of_two(), phase=floats(0, 1)):
    n = draw(ns)
    return fftfit.vonmises_profile(draw(floats(1, 1000)), n, draw(phase)) + (
        1e-3 / n
    ) * np.random.default_rng(0).standard_normal(n)


@composite
def random_templates(draw, ns=powers_of_two()):
    return np.random.randn(draw(ns))


@composite
def boxcar_templates(draw, ns=powers_of_two(), duty=floats(0, 1)):
    n = draw(ns)
    t = np.zeros(n)
    m = int(draw(duty) * n)
    t[:m] = 1
    t[0] = 1
    t[-1] = 0
    return t


@pytest.fixture
def state():
    return np.random.default_rng(0)


def randomized_test(tries=3, seed=0):
    if tries < 1:
        raise ValueError("Must carry out at least one try")

    def rt(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            kwargs.pop("state", None)
            bad_seeds = []
            bad_seed = None
            bad_exc = None
            for i in range(seed, seed + tries):
                try:
                    return f(*args, state=np.random.default_rng(seed), **kwargs)
                except AssertionError as e:
                    bad_seeds.append(i)
                    bad_seed = i
                    bad_exc = e
            raise AssertionError(
                "Test failed for all seeds (%s). Failure for seed %d shown above."
                % (bad_seeds, bad_seed)
            ) from bad_exc

        return wrapper

    return rt


@randomized_test(tries=3)
def test_normal_fpp(state):
    assert state.standard_normal() < 2


@given(
    arrays(complex, integers(3, 9), elements=complex_numbers(max_magnitude=1e8)),
    integers(4, 16),
)
def test_irfft_value(c, n):
    assume(n >= 2 * (len(c) - 1))
    c = c.copy()
    c[0] = c[0].real
    c[-1] = 0
    xs = np.linspace(0, 1, n, endpoint=False)
    assert_rms_close(np.fft.irfft(c, n), fftfit.irfft_value(c, xs, n))


@given(
    arrays(complex, integers(3, 1025), elements=complex_numbers(max_magnitude=1e8)),
    integers(4, 4096),
    floats(0, 1),
)
def test_irfft_value_one(c, n, x):
    assume(n >= 2 * (len(c) - 1))
    fftfit.irfft_value(c, x, n)


@given(floats(0, 1), one_of(vonmises_templates_noisy(), random_templates()))
def test_shift_invertible(s, template):
    assert_allclose(template, fftfit.shift(fftfit.shift(template, s), -s), atol=1e-14)


@given(integers(0, 2 ** 20), floats(1, 1000), integers(5, 16), floats(0, 1))
@pytest.mark.parametrize(
    "fftfit_basic", [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
)
def test_fftfit_basic_integer_vonmises(fftfit_basic, i, kappa, profile_length, phase):
    n = 2 ** profile_length
    template = fftfit.vonmises_profile(kappa, n, phase) + (
        1e-3 / n
    ) * np.random.default_rng(0).standard_normal(n)
    assume(sum(template > 0.5 * template.max()) > 1)
    s = i / len(template)
    rs = fftfit_basic(template, fftfit.shift(template, s))
    assert_allclose_phase(i / len(template), rs)


@given(integers(0, 2 ** 20), vonmises_templates_noisy())
@pytest.mark.parametrize(
    "fftfit_basic", [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
)
def test_fftfit_basic_integer(fftfit_basic, i, template):
    assume(len(template) >= 32)
    s = i / len(template)
    rs = fftfit_basic(template, fftfit.shift(template, s))
    assert_allclose_phase(i / len(template), rs)


@given(integers(0, 2 ** 5), vonmises_templates_noisy())
@pytest.mark.parametrize(
    "fftfit_basic", [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
)
def test_fftfit_basic_integer_fraction(fftfit_basic, i, template):
    s = i / len(template) / 2 ** 5
    rs = fftfit_basic(template, fftfit.shift(template, s))
    assert_allclose_phase(rs, s, atol=1e-4 / len(template))


@given(floats(0, 1), floats(1, 1000), powers_of_two())
@pytest.mark.parametrize(
    "fftfit_basic", [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
)
def test_fftfit_basic_subbin(fftfit_basic, s, kappa, n):
    assume(n >= 32)
    template = fftfit.vonmises_profile(kappa, n) + (1e-3 / n) * np.random.default_rng(
        0
    ).standard_normal(n)
    rs = fftfit_basic(template, fftfit.shift(template, s / n))
    assert_allclose_phase(rs, s / n, atol=1e-4 / len(template))


@given(
    floats(0, 1),
    one_of(vonmises_templates_noisy(), random_templates(), boxcar_templates()),
)
@pytest.mark.parametrize(
    "fftfit_basic", [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
)
def test_fftfit_basic_template(fftfit_basic, s, template):
    assume(len(template) >= 32)
    rs = fftfit_basic(template, fftfit.shift(template, s))
    assert_allclose_phase(rs, s, atol=1e-3 / len(template))


@given(
    one_of(vonmises_templates(), random_templates(), boxcar_templates()),
    one_of(vonmises_templates(), random_templates(), boxcar_templates()),
)
@pytest.mark.parametrize(
    "fftfit_basic", [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
)
def test_fftfit_basic_different_profiles(fftfit_basic, profile1, profile2):
    assume(len(profile1) >= 32)
    fftfit_basic(profile1, profile2)


@given(
    one_of(vonmises_templates(), random_templates()),
    one_of(vonmises_templates(), random_templates()),
)
@pytest.mark.parametrize(
    "fftfit_basic", [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
)
def test_fftfit_basic_shift(fftfit_basic, profile1, profile2):
    assume(len(profile1) >= 32)
    s = fftfit.fftfit_basic(profile1, profile2)
    assert_allclose_phase(
        fftfit.fftfit_basic(fftfit.shift(profile1, s), profile2),
        0,
        atol=1e-3 / min(len(profile1), len(profile2)),
    )


@given(
    one_of(vonmises_templates(), random_templates(), boxcar_templates()),
    floats(0, 1),
    one_of(just(1.0), floats(0.5, 2), floats(1e-5, 1e5)),
    one_of(just(0.0), floats(-1, 1), floats(-1e5, 1e5)),
)
def test_fftfit_compute_scale(template, s, a, b):
    profile = a * fftfit.shift(template, s) + b
    r = fftfit.fftfit_full(template, profile)
    assert_allclose_phase(s, r.shift, atol=1e-3 / len(template))
    assert_allclose(b, r.offset, atol=a * 1e-8)
    assert_allclose(a, r.scale, atol=(1 + abs(b)) * 1e-8)
    assert_rms_close(
        profile, r.scale * fftfit.shift(template, r.shift) + r.offset, atol=1e-7
    )


@pytest.mark.parametrize("kappa,n,std", [(10, 64, 0.01), (100, 1024, 0.02)])
@randomized_test()
def test_fftfit_uncertainty_template(kappa, n, std, state):
    template = fftfit.vonmises_profile(kappa, n)
    r = fftfit.fftfit_full(template, template, std=std)

    def gen_shift():
        return fftfit.wrap(
            fftfit.fftfit_basic(
                template, template + std * state.standard_normal((len(template),))
            )
        )

    values = [gen_shift() for i in range(100)]
    ks, fpp = scipy.stats.kstest(values, scipy.stats.norm(0, r.uncertainty).cdf)


@pytest.mark.parametrize(
    "kappa,n,std,shift,scale,offset",
    [
        (1, 256, 0.01, 0, 1, 0),
        # (10, 64, 0.01, 1 / 3, 2, 0),
        # (100, 1024, 0.02, 0.2, 0.1, 100),
    ],
)
@randomized_test()
def test_fftfit_uncertainty_estimate_std(kappa, n, std, shift, scale, offset, state):
    template = fftfit.vonmises_profile(kappa, n)
    profile = (
        scale * fftfit.shift(template, shift)
        + offset
        # + std * state.standard_normal((len(template),))
    )
    r = fftfit.fftfit_full(template, profile, std=std)

    def gen_shift():
        profile = (
            scale * fftfit.shift(template, shift)
            + offset
            + std * state.standard_normal((len(template),))
        )
        return fftfit.fftfit_basic(template, profile)

    values = np.array([gen_shift() for i in range(100)])
    c = np.sum(np.abs(fftfit.wrap(values - shift)) < r.uncertainty)
    # breakpoint()
    p = 1 - 2 * scipy.stats.norm().sf(1)
    assert (
        scipy.stats.binom.isf(0.01, len(values), p)
        <= c
        <= scipy.stats.binom.isf(0.99, len(values), p)
    )
