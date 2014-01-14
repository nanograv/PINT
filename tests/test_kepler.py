import astropy.units as u
import pint.orbital.kepler as kepler
from pint import ls
from numpy.testing import assert_allclose

def test_mass_solar():
    a = u.au.to(ls)
    pb = u.year.to(u.day)
    m = kepler.mass(a,pb)
    assert_allclose(m,1,rtol=1e-4) # Earth+Sun mass
