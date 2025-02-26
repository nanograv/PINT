from copy import deepcopy

import pytest
from pint.models import get_model, get_model_and_toas
from pint.models.chromatic_model import ChromaticCM
from pint.models.timing_model import MissingParameter
from pint.simulation import make_fake_toas_uniform
from pint.fitter import WLSFitter
import astropy.units as u
from io import StringIO
import numpy as np
from pinttestdata import datadir


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
    assert "TNCHROMIDX" in m and m.TNCHROMIDX.value == 4
    assert len(m.get_CM_terms()) == 3
    assert u.Unit(m.CM_derivative_unit(1)) == m.CM1.units

    assert "CM2" in str(m)

    # test bad model
    with pytest.raises(MissingParameter):
        par = """
            F0 100 1
            CM 0.01
            CM1 0.001
            CM2 0.0001 
        """
        m1 = get_model(StringIO(par))


def test_change_cmepoch():
    par = """
        F0 100 1
        CMEPOCH 55000
        CM 0.01
        CM1 0.001
        CM2 0.0001 
    """
    m0 = get_model(StringIO(par))
    m1 = deepcopy(m0)

    m1.change_cmepoch(55100)
    assert m1.CMEPOCH.value == 55100
    assert np.isclose(
        m1.CM.value,
        (
            m0.CM.quantity
            + m0.CM1.quantity * (100 * u.day)
            + 0.5 * m0.CM2.quantity * (100 * u.day) ** 2
        ).value,
    )


def test_chromatic_cm_fit():
    par = """
        RAJ     2:00:00.00000000
        DECJ    20:00:00.00000000
        F0      100 1 
        PEPOCH  55000
        DM      10 0
        CM      10 1
        CM1     0.001 1
        TNCHROMIDX   4 
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
        np.abs(ftr.model.CM1.value - m.CM1.value) / ftr.model.CM1.uncertainty_value < 3
    )


def test_cm_dm_comparison():
    m, t = get_model_and_toas(datadir / "NGC6440E.par", datadir / "NGC6440E.tim")

    ftr = WLSFitter(t, m)
    ftr.fit_toas(maxiter=3)
    m0 = ftr.model

    dmval = m0.DM.value
    m1 = deepcopy(m0)
    m1.remove_component("DispersionDM")
    m1.add_component(ChromaticCM())
    m1.TNCHROMIDX.value = 2
    m1.CM.value = dmval
    m1.CM.frozen = False

    ftr = WLSFitter(t, m1)
    ftr.fit_toas(maxiter=3)

    assert ftr.resids.chi2_reduced < 1.5
    assert (ftr.model.CM.value - m0.DM.value) / ftr.model.CM.uncertainty_value < 1.1
    assert np.isclose(ftr.model.CM.uncertainty_value, m0.DM.uncertainty_value, atol=0.1)
