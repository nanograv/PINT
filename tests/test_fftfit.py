from functools import wraps
from itertools import product

import numpy as np
import pytest
import scipy.stats
from hypothesis import assume, given, target
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
from pint.profile import fftfit_presto
from pint.profile import fftfit_full, fftfit_basic


fftfit_basics = [fftfit_aarchiba.fftfit_basic, fftfit_nustar.fftfit_basic]
fftfit_fulls = [fftfit_aarchiba.fftfit_full, fftfit_nustar.fftfit_full]

if fftfit_presto.presto is not None:
    fftfit_basics.append(fftfit_presto.fftfit_basic)
    fftfit_fulls.append(fftfit_presto.fftfit_full)

NO_PRESTO = fftfit_presto.presto is None


def assert_rms_close(a, b, rtol=1e-8, atol=1e-8, name=None):
    __tracebackhide__ = True
    target(np.mean((a - b) ** 2), label="mean")
    if name is not None:
        target((a - b).max(), label="{} max".format(name))
        target(-(a - b).min(), label="{} min".format(name))
    assert np.mean((a - b) ** 2) < rtol * (np.mean(a ** 2) + np.mean(b ** 2)) + atol


def assert_allclose_phase(a, b, atol=1e-8, name=None):
    __tracebackhide__ = True
    if name is not None:
        target(np.abs(fftfit.wrap(a - b)).max(), label="{} max".format(name))
        target(np.abs(fftfit.wrap(a - b)).mean(), label="{} mean".format(name))
    assert np.all(np.abs(fftfit.wrap(a - b)) <= atol)


ONE_SIGMA = 1 - 2 * scipy.stats.norm().sf(1)


def assert_happens_with_probability(
    func,
    p=ONE_SIGMA,
    n=100,
    p_lower=None,
    p_upper=None,
    fpp=0.05,
):
    __tracebackhide__ = True
    if p_lower is None:
        p_lower = p
    if p_upper is None:
        p_upper = p
    if p_lower > p_upper:
        raise ValueError(
            "Lower limit on probability {} is higher than upper limit {}".format(
                p_lower, p_upper
            )
        )

    k = 0
    for i in range(n):
        if func():
            k += 1
    low_k = scipy.stats.binom(n, p_lower).ppf(fpp / 2)
    high_k = scipy.stats.binom(n, p_upper).isf(fpp / 2)
    assert low_k <= k
    assert k <= high_k


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


def randomized_test(tries=5, seed=0):
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
    "code",
    [
        "aarchiba",
        pytest.param(
            "nustar",
            marks=[
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
        pytest.param(
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
    ],
)
def test_fftfit_basic_integer_vonmises(code, i, kappa, profile_length, phase):
    if code == "presto":
        assume(profile_length <= 13)
    n = 2 ** profile_length
    template = fftfit.vonmises_profile(kappa, n, phase) + (
        1e-3 / n
    ) * np.random.default_rng(0).standard_normal(n)
    assume(sum(template > 0.5 * template.max()) > 1)
    s = i / len(template)
    rs = fftfit_basic(template, fftfit.shift(template, s), code=code)
    assert_allclose_phase(s, rs, atol=1 / (32 * len(template)), name="shift")


@given(integers(0, 2 ** 20), vonmises_templates_noisy())
@pytest.mark.parametrize(
    "code",
    [
        "aarchiba",
        pytest.param(
            "nustar",
            marks=[
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
        pytest.param(
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
    ],
)
def test_fftfit_basic_integer(code, i, template):
    if code != "aarchiba":
        assume(len(template) >= 32)
    s = i / len(template)
    rs = fftfit_basic(template, fftfit.shift(template, s), code=code)
    assert_allclose_phase(s, rs, name="shift")


@given(integers(0, 2 ** 5), vonmises_templates_noisy())
@pytest.mark.parametrize(
    "code",
    [
        "aarchiba",
        pytest.param(
            "nustar",
            marks=[
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
        pytest.param(
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
    ],
)
def test_fftfit_basic_integer_fraction(code, i, template):
    s = i / len(template) / 2 ** 5
    rs = fftfit_basic(template, fftfit.shift(template, s), code=code)
    assert_allclose_phase(rs, s, atol=1e-4 / len(template), name="shift")


@given(floats(0, 1), floats(1, 1000), powers_of_two())
@pytest.mark.parametrize(
    "code",
    [
        "aarchiba",
        pytest.param(
            "nustar",
            marks=[
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
        pytest.param(
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
    ],
)
def test_fftfit_basic_subbin(code, s, kappa, n):
    if code != "aarchiba":
        assume(n >= 32)
    template = fftfit.vonmises_profile(kappa, n) + (1e-3 / n) * np.random.default_rng(
        0
    ).standard_normal(n)
    rs = fftfit_basic(template, fftfit.shift(template, s / n), code=code)
    assert_allclose_phase(rs, s / n, atol=1e-4 / len(template), name="shift")


@given(
    floats(0, 1),
    one_of(vonmises_templates_noisy(), random_templates(), boxcar_templates()),
)
@pytest.mark.parametrize(
    "code",
    [
        "aarchiba",
        pytest.param(
            "nustar",
            marks=[
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
        pytest.param(
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
                pytest.mark.xfail(reason="profile too symmetric"),
            ],
        ),
    ],
)
def test_fftfit_basic_template(code, s, template):
    if code != "aarchiba":
        assume(len(template) >= 32)
    rs = fftfit_basic(template, fftfit.shift(template, s), code=code)
    assert_allclose_phase(rs, s, atol=1e-3 / len(template), name="shift")


@given(
    one_of(vonmises_templates(), random_templates(), boxcar_templates()),
    one_of(vonmises_templates(), random_templates(), boxcar_templates()),
)
@pytest.mark.parametrize(
    "code",
    [
        "aarchiba",
        pytest.param(
            "nustar",
            marks=[
                pytest.mark.xfail(reason="profiles different lengths"),
            ],
        ),
        pytest.param(
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
                pytest.mark.xfail(reason="profiles different lengths"),
            ],
        ),
    ],
)
def test_fftfit_basic_different_profiles(code, profile1, profile2):
    if code != "aarchiba":
        assume(len(profile1) >= 32)
    fftfit_basic(profile1, profile2, code=code)


@given(
    one_of(vonmises_templates(), random_templates()),
    one_of(vonmises_templates(), random_templates()),
)
@pytest.mark.parametrize(
    "code",
    [
        "aarchiba",
        pytest.param(
            "nustar",
            marks=[
                pytest.mark.xfail(reason="profiles different lengths"),
            ],
        ),
        pytest.param(
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
                pytest.mark.xfail(reason="profiles different lengths"),
            ],
        ),
    ],
)
def test_fftfit_shift_equivalence(code, profile1, profile2):
    if code != "aarchiba":
        assume(len(profile1) >= 32)
    s = fftfit_basic(profile1, profile2, code=code)
    assert_allclose_phase(
        fftfit_basic(fftfit.shift(profile1, s), profile2, code=code),
        0,
        atol=1e-3 / min(len(profile1), len(profile2)),
        name="shift",
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
    assert_allclose_phase(s, r.shift, atol=1e-3 / len(template), name="shift")
    assert_allclose(b, r.offset, atol=a * 1e-8)
    assert_allclose(a, r.scale, atol=(1 + abs(b)) * 1e-8)
    assert_rms_close(
        profile,
        r.scale * fftfit.shift(template, r.shift) + r.offset,
        atol=1e-7,
        name="profile",
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


# could be hypothesized
@pytest.mark.parametrize(
    "kappa,n,std,shift,scale,offset",
    [
        (1, 256, 0.01, 0, 1, 0),
        (10, 64, 0.01, 1 / 3, 2e-3, 0),
        (100, 1024, 0.02, 0.2, 1e4, 0),
        (100, 2048, 0.01, 0.2, 1e4, -100),
    ],
)
def test_fftfit_uncertainty_scaling_invariance(kappa, n, std, shift, scale, offset):
    state = np.random.default_rng(0)
    template = fftfit.vonmises_profile(kappa, n)
    profile = fftfit.shift(template, shift) + std * state.standard_normal(len(template))

    r_1 = fftfit.fftfit_full(template, profile)
    r_2 = fftfit.fftfit_full(template, scale * profile + offset)

    assert_allclose_phase(r_2.shift, r_1.shift, 1.0 / (32 * n))
    assert_allclose(r_2.uncertainty, r_1.uncertainty, rtol=1e-3)
    assert_allclose(r_2.scale, scale * r_1.scale, rtol=1e-3)
    assert_allclose(r_2.offset, offset + scale * r_1.offset, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kappa,n,std,shift,scale,offset,estimate",
    [
        a + (b,)
        for a, b in product(
            [
                (1, 256, 0.01, 0, 1, 0),
                (10, 64, 0.01, 1 / 3, 1e-6, 0),
                (100, 1024, 0.002, 0.2, 1e4, 0),
                (100, 1024, 0.02, 0.2, 1e4, 0),
            ],
            [False, True],
        )
    ],
)
@randomized_test(tries=8)
def test_fftfit_uncertainty_estimate(
    kappa, n, std, shift, scale, offset, estimate, state
):
    """Check the noise level estimation works."""
    template = fftfit.vonmises_profile(kappa, n)

    def value_within_one_sigma():
        profile = (
            fftfit.shift(template, shift)
            + offset
            + std * state.standard_normal(len(template))
        )
        if estimate:
            r = fftfit.fftfit_full(template, scale * profile)
        else:
            r = fftfit.fftfit_full(template, scale * profile, std=scale * std)
        return np.abs(fftfit.wrap(r.shift - shift)) < r.uncertainty

    assert_happens_with_probability(value_within_one_sigma, ONE_SIGMA)


@pytest.mark.parametrize(
    "kappa,n,std,shift,scale,offset,code",
    [
        (1, 256, 0.01, 0, 1, 0, "aarchiba"),
        (10, 64, 0.01, 1 / 3, 1e-6, 0, "aarchiba"),
        (100, 1024, 0.002, 0.2, 1e4, 0, "aarchiba"),
        (100, 1024, 0.02, 0.2, 1e4, 0, "aarchiba"),
        (1000, 4096, 0.01, 0.7, 1e4, 0, "aarchiba"),
        pytest.param(
            1,
            256,
            0.01,
            0,
            1,
            0,
            "nustar",
        ),
        pytest.param(
            10,
            64,
            0.01,
            1 / 3,
            1e-6,
            0,
            "nustar",
        ),
        pytest.param(
            100,
            1024,
            0.002,
            0.2,
            1e4,
            0,
            "nustar",
        ),
        pytest.param(
            100,
            1024,
            0.02,
            0.2,
            1e4,
            0,
            "nustar",
        ),
        pytest.param(
            1000,
            4096,
            0.01,
            0.7,
            1e4,
            0,
            "nustar",
        ),
        pytest.param(
            1,
            256,
            0.01,
            0,
            1,
            0,
            "presto",
            marks=[
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
            ],
        ),
        pytest.param(
            10,
            64,
            0.01,
            1 / 3,
            1e-6,
            0,
            "presto",
            marks=[
                pytest.mark.xfail(reason="bug?"),
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
            ],
        ),
        pytest.param(
            100,
            1024,
            0.002,
            0.2,
            1e4,
            0,
            "presto",
            marks=[
                pytest.mark.xfail(reason="bug?"),
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
            ],
        ),
        pytest.param(
            100,
            1024,
            0.02,
            0.2,
            1e4,
            0,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available")],
        ),
        pytest.param(
            1000,
            4096,
            0.01,
            0.7,
            1e4,
            0,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available")],
        ),
    ],
)
@randomized_test(tries=8)
def test_fftfit_value(kappa, n, std, shift, scale, offset, code, state):
    """Check if the returned values are okay with a noisy profile.

    Here we define "okay" as scattered about the right value and within
    one sigma as defined by the uncertainty returned by the aarchiba version
    of the code (this is presumably a trusted uncertainty).
    """
    template = fftfit.vonmises_profile(kappa, n)
    profile = (
        fftfit.shift(template, shift)
        + offset
        + std * state.standard_normal(len(template))
    )
    r_true = fftfit.fftfit_full(template, scale * profile, std=scale * std)
    assert r_true.uncertainty < 0.1, "This uncertainty is too big for accuracy"

    def value_within_one_sigma():
        profile = (
            fftfit.shift(template, shift)
            + offset
            + std * state.standard_normal(len(template))
        )
        r = fftfit_full(template, scale * profile, code=code)
        return np.abs(fftfit.wrap(r.shift - shift)) < r_true.uncertainty

    assert_happens_with_probability(value_within_one_sigma, ONE_SIGMA)


@pytest.mark.parametrize(
    "kappa,n,std,shift,scale,offset,code",
    [
        (1, 256, 0.01, 0, 1, 0, "aarchiba"),
        (10, 64, 0.01, 1 / 3, 1e-6, 0, "aarchiba"),
        (100, 1024, 0.002, 0.2, 1e4, 0, "aarchiba"),
        (100, 1024, 0.02, 0.2, 1e4, 0, "aarchiba"),
        (1000, 4096, 0.01, 0.7, 1e4, 0, "aarchiba"),
        pytest.param(
            1,
            256,
            0.01,
            0,
            1,
            0,
            "nustar",
            marks=pytest.mark.xfail(reason="claimed uncertainty too big"),
        ),
        pytest.param(
            10,
            64,
            0.01,
            1 / 3,
            1e-6,
            0,
            "nustar",
            marks=pytest.mark.xfail(reason="bug?"),
        ),
        pytest.param(
            100,
            1024,
            0.002,
            0.2,
            1e4,
            0,
            "nustar",
            marks=pytest.mark.xfail(reason="bug?"),
        ),
        pytest.param(
            100,
            1024,
            0.02,
            0.2,
            1e4,
            0,
            "nustar",
            marks=pytest.mark.xfail(reason="bug?"),
        ),
        pytest.param(
            1000,
            4096,
            0.01,
            0.7,
            1e4,
            0,
            "nustar",
            marks=pytest.mark.xfail(reason="bug?"),
        ),
        pytest.param(
            1,
            256,
            0.01,
            0,
            1,
            0,
            "presto",
            marks=[
                pytest.mark.xfail(reason="bug?"),
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
            ],
        ),
        pytest.param(
            10,
            64,
            0.01,
            1 / 3,
            1e-6,
            0,
            "presto",
            marks=[
                pytest.mark.xfail(reason="bug?"),
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
            ],
        ),
        pytest.param(
            100,
            1024,
            0.002,
            0.2,
            1e4,
            0,
            "presto",
            marks=[
                pytest.mark.xfail(reason="bug?"),
                pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available"),
            ],
        ),
        pytest.param(
            100,
            1024,
            0.02,
            0.2,
            1e4,
            0,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available")],
        ),
        pytest.param(
            1000,
            4096,
            0.01,
            0.7,
            1e4,
            0,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO is not available")],
        ),
    ],
)
@randomized_test(tries=8)
def test_fftfit_value_vs_uncertainty(kappa, n, std, shift, scale, offset, code, state):
    """Check if the scatter matches the claimed uncertainty."""
    template = fftfit.vonmises_profile(kappa, n)

    def value_within_one_sigma():
        profile = (
            fftfit.shift(template, shift)
            + offset
            + std * state.standard_normal(len(template))
        )
        r = fftfit_full(template, scale * profile, code=code)
        assert r.uncertainty < 0.1, "This uncertainty is too big for accuracy"
        return np.abs(fftfit.wrap(r.shift - shift)) < r.uncertainty

    assert_happens_with_probability(value_within_one_sigma, ONE_SIGMA)


@pytest.mark.parametrize(
    "kappa1,kappa2,n,std,code",
    [
        (1, 1.1, 256, 0.01, "aarchiba"),
        (10, 11, 2048, 0.01, "aarchiba"),
        (100, 110, 2048, 0.01, "aarchiba"),
        (1.1, 1, 256, 0.01, "aarchiba"),
        (11, 10, 2048, 0.01, "aarchiba"),
        (110, 100, 2048, 0.01, "aarchiba"),
        (1, 1.1, 256, 0.01, "nustar"),
        (10, 11, 2048, 0.01, "nustar"),
        (100, 110, 2048, 0.01, "nustar"),
        (1.1, 1, 256, 0.01, "nustar"),
        (11, 10, 2048, 0.01, "nustar"),
        (110, 100, 2048, 0.01, "nustar"),
        pytest.param(
            1,
            1.1,
            256,
            0.01,
            "presto",
            marks=[pytest.mark.xfail(reason="bug?"), pytest.mark.skipif(NO_PRESTO, reason="PRESTO not available")],
        ),
        pytest.param(
            10,
            11,
            2048,
            0.01,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO not available")],
        ),
        pytest.param(
            100,
            110,
            2048,
            0.01,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO not available")],
        ),
        pytest.param(
            1.1,
            1,
            256,
            0.01,
            "presto",
            marks=[pytest.mark.xfail(reason="bug?"), pytest.mark.skipif(NO_PRESTO, reason="PRESTO not available")],
        ),
        pytest.param(
            11,
            10,
            2048,
            0.01,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO not available")],
        ),
        pytest.param(
            110,
            100,
            2048,
            0.01,
            "presto",
            marks=[pytest.mark.skipif(NO_PRESTO, reason="PRESTO not available")],
        ),
    ],
)
@randomized_test(tries=8)
def test_fftfit_wrong_profile(kappa1, kappa2, n, std, code, state):
    """Check that the uncertainty is okay or pessimistic if the template is wrong."""
    template = fftfit.vonmises_profile(kappa1, n)
    wrong_template = fftfit.vonmises_profile(kappa2, n)

    def value_within_one_sigma():
        shift = state.uniform(0, 1)
        profile = fftfit.shift(template, shift) + std * state.standard_normal(
            len(template)
        )
        r = fftfit_full(wrong_template, profile, code=code)
        return np.abs(fftfit.wrap(r.shift - shift)) < r.uncertainty

    # Must be pessimistic
    assert_happens_with_probability(value_within_one_sigma, ONE_SIGMA, p_upper=1)
