from numpy.testing import assert_allclose

import pint.eventstats as es
import numpy as np


def test_sig2sigma():
    sig = [1, 1.0e-1, 1.0e-10, 1.0e-16, 1.0e-120]
    res = es.sig2sigma(sig)
    ans = [0, 1.64485363, 6.46695109, 8.30478543, 23.36370742]
    assert_allclose(res, ans, atol=1.0e-7)

    res = es.sig2sigma(np.log(sig), logprob=True)
    assert_allclose(res, ans, atol=1.0e-7)


def test_h_and_z():
    phases = np.linspace(0, 1, 101)
    weights = np.zeros(101)
    weights[:50] = 1

    ans = 0.01980198019801975
    res = es.hm(phases)
    assert_allclose(res, ans, atol=1.0e-7)

    ans = 40.54180942257532
    res = es.hmw(phases, weights)
    assert_allclose(res, ans, atol=1.0e-7)

    res = es.z2m(phases, m=4)
    assert len(res) == 4
    ans = 0.0792079207920788
    assert_allclose(res[3], ans, atol=1.0e-7)

    ans = 40.54180942257532
    res = es.hmw(phases, weights)
    assert_allclose(res, ans, atol=1.0e-7)

    res = es.z2mw(phases, weights, m=4)
    assert len(res) == 4
    ans = 45.05833019383544
    assert_allclose(res[3], ans, atol=1.0e-7)
