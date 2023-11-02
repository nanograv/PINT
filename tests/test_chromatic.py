from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.fitter import WLSFitter
import astropy.units as u
from io import StringIO
import numpy as np


def test_chromatic_cm():
    par = """
        F0 100 1
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


def test_chromatic_cm_fit():
    par = """
        RAJ     2:00:00.00000000
        DECJ    20:00:00.00000000
        F0      100 1 
        PEPOCH  55000
        DM      10 0
        CM      10 1
        CMIDX   4 1
        TZRSITE   gbt
        TZRMJD  55000
        TZRFRQ  500
    """
    m = get_model(StringIO(par))

    freqs = np.linspace(300, 2000, 16) * u.MHz
    t = make_fake_toas_uniform(50000, 55000, 800, m, add_noise=True, freq=freqs)

    ftr = WLSFitter(t, m)
    ftr.fit_toas(maxiter=5)

    assert np.abs(ftr.model.CM.value - m.CM.value) / ftr.model.CM.uncertainty_value < 3
    assert (
        np.abs(ftr.model.CMIDX.value - m.CMIDX.value) / ftr.model.CM.uncertainty_value
        < 3
    )
