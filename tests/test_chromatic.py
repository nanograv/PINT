from copy import deepcopy

import pytest
from pint.models import get_model, get_model_and_toas
from pint.models.chromatic_model import ChromaticCM
from pint.exceptions import MissingParameter
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

    assert ftr.resids.reduced_chi2 < 1.5
    assert (ftr.model.CM.value - m0.DM.value) / ftr.model.CM.uncertainty_value < 1.1
    assert np.isclose(ftr.model.CM.uncertainty_value, m0.DM.uncertainty_value, atol=0.1)


def test_cm_fref_default():
    """CM_FREF defaults to 1 MHz and is written to the par file."""
    par = """
        F0 100 1
        CM 0.01
    """
    m = get_model(StringIO(par))

    assert "CM_FREF" in m
    assert m.CM_FREF.quantity == 1 * u.MHz
    assert m.CM_FREF.frozen
    assert "CM_FREF" in str(m)

    m2 = get_model(StringIO(str(m)))
    assert m2.CM_FREF.quantity == m.CM_FREF.quantity


def test_cm_fref_from_parfile():
    par = """
        F0 100 1
        CM 0.01
        CM_FREF 800
    """
    m = get_model(StringIO(par))
    assert m.CM_FREF.quantity == 800 * u.MHz


@pytest.mark.parametrize("fref", [1.0, 800.0, 1400.0, 2000.0])
def test_cm_fref_delay_scaling(fref):
    """The delay is CM * DMconst * (freq / CM_FREF)**(-TNCHROMIDX)."""
    from pint import DMconst

    par = f"""
        PSR J1234+5678
        RAJ 12:34:00
        DECJ 56:12:00
        F0 100 1
        PEPOCH 55000
        CM 0.01
        TNCHROMIDX 4
        CM_FREF {fref}
    """
    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(
        55000, 55500, 8, m, freq=np.array([800.0, 1400.0, 2000.0]) * u.MHz
    )

    delay = m.components["ChromaticCM"].constant_chromatic_delay(t)
    freqs = m.barycentric_radio_freq(t)
    expected = (
        DMconst * m.CM.quantity * (freqs / m.CM_FREF.quantity) ** -m.TNCHROMIDX.value
    ).to(u.s)

    assert np.allclose(delay.to_value(u.s), expected.to_value(u.s))


def test_chromatic_time_delay_mixed_frequency_units():
    """The frequency ratio is made dimensionless, so freq and fref may use
    different (compatible) units; incompatible units raise."""
    m = get_model(StringIO("F0 100 1\nCM 0.01\nTNCHROMIDX 4\n"))
    comp = m.components["ChromaticCM"]
    freqs = np.array([800.0, 1400.0, 2000.0]) * u.MHz

    d_mhz = comp.chromatic_time_delay(m.CM.quantity, 4.0, freqs, 1400 * u.MHz)
    d_ghz = comp.chromatic_time_delay(m.CM.quantity, 4.0, freqs.to(u.GHz), 1.4 * u.GHz)
    d_mix = comp.chromatic_time_delay(m.CM.quantity, 4.0, freqs.to(u.GHz), 1400 * u.MHz)

    assert d_mhz.unit == u.s
    assert np.allclose(d_ghz.to_value(u.s), d_mhz.to_value(u.s))
    assert np.allclose(d_mix.to_value(u.s), d_mhz.to_value(u.s))
    with pytest.raises(u.UnitConversionError):
        comp.chromatic_time_delay(m.CM.quantity, 4.0, freqs, 1 * u.s)


def test_cm_fref_rescales_delay():
    """Halving CM_FREF scales the delay by (1/2)**TNCHROMIDX."""
    par = """
        PSR J1234+5678
        RAJ 12:34:00
        DECJ 56:12:00
        F0 100 1
        PEPOCH 55000
        CM 0.01
        TNCHROMIDX 4
        CM_FREF {fref}
    """
    m1 = get_model(StringIO(par.format(fref=1400)))
    m2 = get_model(StringIO(par.format(fref=700)))
    t = make_fake_toas_uniform(55000, 55500, 8, m1, freq=1400 * u.MHz)

    d1 = m1.components["ChromaticCM"].constant_chromatic_delay(t)
    d2 = m2.components["ChromaticCM"].constant_chromatic_delay(t)

    assert np.allclose(
        (d2 / d1).to_value(u.dimensionless_unscaled), 0.5**m1.TNCHROMIDX.value
    )


def test_cm_fref_fit_is_consistent():
    """Fitting CM recovers the injected value for a non-default CM_FREF."""
    par = """
        RAJ     2:00:00.00000000
        DECJ    20:00:00.00000000
        F0      100 1
        PEPOCH  55000
        DM      10 0
        CM      1e-3 1
        TNCHROMIDX   4
        CM_FREF 800
        TZRSITE   gbt
        TZRMJD  55000
        TZRFRQ  500
    """
    m = get_model(StringIO(par))
    freqs = np.linspace(300, 2000, 16) * u.MHz
    t = make_fake_toas_uniform(50000, 55000, 800, m, add_noise=True, freq=freqs)

    ftr = WLSFitter(t, m)
    ftr.fit_toas(maxiter=5)

    assert ftr.model.CM_FREF.quantity == 800 * u.MHz
    assert np.abs(ftr.model.CM.value - m.CM.value) / ftr.model.CM.uncertainty_value < 3
