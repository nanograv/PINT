from pint.models import get_model
import astropy.units as u
from io import StringIO


def test_chromatic_cm():
    par = """F0 100 1
    CMEPOCH 55000
    CM 0.01
    CM1 0.001
    CM2 0.0001 
    """
    m = get_model(StringIO(par))

    assert "ChromaticCM" in m.components
    assert "CM" in m
    assert "CM1" in m
    assert "CMEPOCH" in m
    assert "CMIDX" in m and m.CMIDX.value == 4
    assert len(m.get_CM_terms()) == 3
    assert u.Unit(m.CM_derivative_unit(1)) == m.CM1.units
