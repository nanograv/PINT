import astropy.units as u
import pint.orbital.kepler as kepler
from pint import ls
from numpy.testing import assert_allclose
from pint.utils import check_all_partials

def test_mass_solar():
    a = u.au.to(ls)
    pb = u.year.to(u.day)
    m = kepler.mass(a,pb)
    assert_allclose(m,1,rtol=1e-4) # Earth+Sun mass

def test_kepler_2d_t0():
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.2, eps2=0.1, t0=1)
    xyv, _ = kepler.kepler_2d(p, p.t0)
    assert xyv[0]>0
    assert_allclose(xyv[1],0,atol=1e-8)

def test_kepler_2d_t0_pb():
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.1, eps2=0.2, t0=1)
    xyv, _ = kepler.kepler_2d(p, p.t0+p.pb)
    assert xyv[0]>0
    assert_allclose(xyv[1],0,atol=1e-8)


def flatten_namedtuple(f,tupletype):
    l = len(tupletype._fields)
    def new_f(*args):
        new_args = [tupletype(*args[:l])] + list(args[l:])
        return f(*new_args)
    return new_f

def test_kepler_2d_derivs():
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.2, eps2=0.1, t0=-1)
    check_all_partials(
        flatten_namedtuple(kepler.kepler_2d,kepler.Kepler2DParameters),
                       list(p)+[4])

def test_kepler_2d_inverse():
    t = 23
    p = kepler.Kepler2DParameters(a=2, pb=3, eps1=0.2, eps2=0.1, t0=t-1)
    xyv, _ = kepler.kepler_2d(p,t)
    m = kepler.mass(p.a, p.pb)
    p2 = kepler.inverse_kepler_2d(xyv, m, t)

    assert_allclose(p,p2,atol=1e-8)

