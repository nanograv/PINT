import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose

import pint.orbital.kepler as kepler
from pint import ls
from pint.utils import check_all_partials


def test_mass_solar():
    a = u.au.to(ls)
    pb = u.year.to(u.day)
    m = kepler.mass(a, pb)
    assert_allclose(m, 1, rtol=1e-4)  # Earth+Sun mass


def test_mass_derivs():
    a = 2
    pb = 3
    check_all_partials(kepler.mass_partials, [a, pb])


def test_kepler_2d_t0():
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.2, eps2=0.1, t0=1)
    xyv, _ = kepler.kepler_2d(p, p.t0)
    assert xyv[0] > 0
    assert_allclose(xyv[1], 0, atol=1e-8)


def test_kepler_2d_t0_pb():
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.1, eps2=0.2, t0=1)
    xyv, _ = kepler.kepler_2d(p, p.t0 + p.pb)
    assert xyv[0] > 0
    assert_allclose(xyv[1], 0, atol=1e-8)


def test_kepler_2d_circ():
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0, eps2=0, t0=1)
    xyv, partials = kepler.kepler_2d(p, p.t0)
    assert xyv[0] > 0
    assert_allclose(xyv[1], 0, atol=1e-8)
    assert np.all(np.isfinite(xyv))
    assert np.all(np.isfinite(partials))

    xyv, partials = kepler.kepler_2d(p, 0)
    assert np.all(np.isfinite(xyv))
    assert np.all(np.isfinite(partials))


def flatten_namedtuple(f, tupletype):
    l = len(tupletype._fields)

    def new_f(*args):
        new_args = [tupletype(*args[:l])] + list(args[l:])
        return f(*new_args)

    return new_f


def test_kepler_2d_derivs():
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.2, eps2=0.1, t0=-1)
    check_all_partials(
        flatten_namedtuple(kepler.kepler_2d, kepler.Kepler2DParameters), list(p) + [4]
    )


def test_kepler_2d_inverse():
    t = 23
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.2, eps2=0.1, t0=t - 1)
    xyv, _ = kepler.kepler_2d(p, t)
    m = kepler.mass(p.a, p.pb)
    p2 = kepler.inverse_kepler_2d(xyv, m, t)

    assert_allclose(p, p2, atol=1e-8)


def test_kepler_3d_derivs():
    p = kepler.Kepler3DParameters(
        a=2, pb=3, eps1=0.2, eps2=0.1, i=np.pi / 3, lan=np.pi / 4, t0=-1
    )
    check_all_partials(
        flatten_namedtuple(kepler.kepler_3d, kepler.Kepler3DParameters), list(p) + [4]
    )


def test_kepler_3d_inverse():
    t = 23
    p = kepler.Kepler3DParameters(
        a=2, pb=3, eps1=0.2, eps2=0.1, i=np.pi / 3, lan=np.pi / 4, t0=t - 1
    )
    xyv, _ = kepler.kepler_3d(p, t)
    m = kepler.mass(p.a, p.pb)
    p2 = kepler.inverse_kepler_3d(xyv, m, t)

    assert_allclose(p, p2, atol=1e-8)


def test_kepler_two_body_derivs():
    p = kepler.KeplerTwoBodyParameters(
        a=2,
        pb=3,
        eps1=0.2,
        eps2=0.1,
        i=np.pi / 3,
        lan=np.pi / 4,
        q=0.2,
        x_cm=0,
        y_cm=0,
        z_cm=0,
        vx_cm=0,
        vy_cm=0,
        vz_cm=0,
        tasc=-1,
    )
    check_all_partials(
        flatten_namedtuple(kepler.kepler_two_body, kepler.KeplerTwoBodyParameters),
        list(p) + [4],
    )


def test_kepler_two_body_inverse():
    t = 12
    p = kepler.KeplerTwoBodyParameters(
        a=2,
        pb=3,
        eps1=0.2,
        eps2=0.1,
        i=np.pi / 3,
        lan=np.pi / 4,
        q=0.2,
        x_cm=0,
        y_cm=0,
        z_cm=0,
        vx_cm=0,
        vy_cm=0,
        vz_cm=0,
        tasc=t - 1,
    )
    xvm, _ = kepler.kepler_two_body(p, t)
    p2 = kepler.inverse_kepler_two_body(xvm, t)

    assert_allclose(p, p2, atol=1e-8)


def test_btx_parameters():
    asini = 1
    pb = 1
    eps1 = 0.001
    eps2 = 0.002
    tasc = 0

    asini, pb, e, om, t0 = kepler.btx_parameters(asini, pb, eps1, eps2, tasc)

    assert np.isclose(e**2, eps1**2 + eps2**2)
    assert np.isclose(np.tan(om), eps1 / eps2)
    assert np.isfinite(t0)
