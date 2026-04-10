from io import StringIO
import numpy as np
import pytest
import astropy.units as u

from pint.fitter import Fitter
from pint.models.model_builder import get_model
from pint.models.transient_events import ChromaticGaussianEvent
from pint.simulation import make_fake_toas_uniform
from pint.residuals import Residuals


@pytest.fixture
def model_and_toas():
    par = """
        RAJ            05:00:00                     1
        DECJ           15:00:00                     1
        POSEPOCH       55000
        F0             100                          1
        F1             -1e-15                       1
        PEPOCH         55000
        EXPDIPEP_1     54764.272428904194001        1      
        EXPDIPAMP_1    1.6641670367524487e-06       1
        EXPDIPTAU_1    112.00425959054773           1
        EXPDIPIDX_1    -1.9148109887274356          1
        EXPEP_2        55764.272428904194001        1      
        EXPPH_2        2.6641670367524487e-06       1
        EXPTAU_2       102.00425959054773           1
        EXPINDEX_2     -2.9148109887274356          1
        EXPDIPEPS      0.01 
        TZRMJD         55000
        TZRSITE        pks
        TZRFRQ         1400
        EPHEM          DE440
        CLOCK          TT(BIPM2019)
        UNITS          TDB
    """
    m = get_model(StringIO(par))

    freqs = np.linspace(500, 1500, 8) * u.MHz
    t = make_fake_toas_uniform(
        54000,
        58000,
        4000,
        m,
        freq=freqs,
        obs="pks",
        add_noise=True,
        multi_freqs_in_epoch=True,
    )

    return m, t


def test_expdip(model_and_toas):
    m, t = model_and_toas

    assert len(m.components["SimpleExponentialDip"].get_indices()) == 2

    res = Residuals(t, m)
    assert res.reduced_chi2 < 1.5

    ftr = Fitter.auto(t, m)
    ftr.fit_toas(maxiter=10)
    assert ftr.resids.reduced_chi2 < 1.5

    for p in m.free_params:
        assert (m[p].value - ftr.model[p].value) / ftr.model[p].uncertainty_value < 3


# ============================================================
# ChromaticGaussianEvent tests
# ============================================================

BASE_PAR = """
    RAJ            05:00:00                     1
    DECJ           15:00:00                     1
    POSEPOCH       55000
    F0             100                          1
    F1             -1e-15                       1
    PEPOCH         55000
    TZRMJD         55000
    TZRSITE        @
    TZRFRQ         1400
    EPHEM          DE440
    CLOCK          TT(BIPM2019)
    UNITS          TDB
"""


@pytest.fixture
def base_model():
    """A minimal timing model without any chromatic Gaussian events."""
    return get_model(StringIO(BASE_PAR))


@pytest.fixture
def base_toas(base_model):
    """Fake TOAs spanning the range where events will be injected."""
    freqs = np.linspace(500, 1500, 8) * u.MHz
    return make_fake_toas_uniform(
        54000,
        58000,
        4000,
        base_model,
        freq=freqs,
        obs="@",
        add_noise=True,
        multi_freqs_in_epoch=True,
    )


def _add_chromgauss_component(model):
    """Helper to add a ChromaticGaussianEvent component to a model."""
    comp = ChromaticGaussianEvent()
    model.add_component(comp)
    comp.remove_chrom_gauss_event(1)
    return model


def test_chromgauss_add_one(base_model, base_toas):
    """Add a single chromatic Gaussian event and verify parameters and delay."""
    m = _add_chromgauss_component(base_model)
    comp = m.components["ChromaticGaussianEvent"]

    idx = comp.add_chromatic_gaussian_event(
        epoch=55500.0,
        log10amp=-6.0,
        chromidx=2.0,
        log10sigma=1.5,
        sign=1,
        frozen=True,
    )

    assert idx == 1
    assert len(comp.get_indices()) == 1
    assert 1 in comp.get_indices()

    # All five prefix parameters should be present
    assert hasattr(comp, "CHROMGAUSS_EPOCH_1")
    assert hasattr(comp, "CHROMGAUSS_LOGAMP_1")
    assert hasattr(comp, "CHROMGAUSS_CHROMIDX_1")
    assert hasattr(comp, "CHROMGAUSS_LOGSIG_1")
    assert hasattr(comp, "CHROMGAUSS_SIGN_1")

    # Delay computation should not raise
    delay = comp.chrom_gauss_delay(base_toas)
    assert delay.unit == u.s
    assert len(delay) == len(base_toas)

    # Residuals should be finite
    res = Residuals(base_toas, m)
    assert np.all(np.isfinite(res.calc_time_resids().value))


def test_chromgauss_add_two(base_model, base_toas):
    """Add two chromatic Gaussian events and verify both are tracked."""
    m = _add_chromgauss_component(base_model)
    comp = m.components["ChromaticGaussianEvent"]

    idx1 = comp.add_chromatic_gaussian_event(
        epoch=55000.0,
        log10amp=-6.0,
        chromidx=2.0,
        log10sigma=1.5,
        sign=1,
    )
    idx2 = comp.add_chromatic_gaussian_event(
        epoch=56000.0,
        log10amp=-5.5,
        chromidx=2.0,
        log10sigma=1.3,
        sign=-1,
    )

    assert idx1 == 1
    assert idx2 == 2
    assert len(comp.get_indices()) == 2
    assert set(comp.get_indices()) == {1, 2}

    # Delay should be the sum of both events
    delay = comp.chrom_gauss_delay(base_toas)
    assert delay.unit == u.s

    res = Residuals(base_toas, m)
    assert np.all(np.isfinite(res.calc_time_resids().value))


def test_chromgauss_remove(base_model, base_toas):
    """Add two events, remove one, then remove the other."""
    m = _add_chromgauss_component(base_model)
    comp = m.components["ChromaticGaussianEvent"]

    comp.add_chromatic_gaussian_event(
        epoch=55000.0, log10amp=-6.0, chromidx=2.0, log10sigma=1.5, sign=1
    )
    comp.add_chromatic_gaussian_event(
        epoch=56000.0, log10amp=-5.5, chromidx=2.0, log10sigma=1.3, sign=-1
    )
    assert len(comp.get_indices()) == 2

    # Remove event 1
    comp.remove_chrom_gauss_event(1)
    assert len(comp.get_indices()) == 1
    assert 2 in comp.get_indices()
    assert 1 not in comp.get_indices()
    assert not hasattr(comp, "CHROMGAUSS_EPOCH_1")

    # Residuals should still work with one event
    res = Residuals(base_toas, m)
    assert np.all(np.isfinite(res.calc_time_resids().value))

    # Remove event 2
    comp.remove_chrom_gauss_event(2)
    assert len(comp.get_indices()) == 0

    # Delay should be zero with no events
    delay = comp.chrom_gauss_delay(base_toas)
    assert np.allclose(delay.value, 0.0)


def test_chromgauss_remove_multiple(base_model):
    """Remove multiple events at once using a list of indices."""
    m = _add_chromgauss_component(base_model)
    comp = m.components["ChromaticGaussianEvent"]

    for i in range(3):
        comp.add_chromatic_gaussian_event(
            epoch=55000.0 + i * 500,
            log10amp=-6.0,
            chromidx=2.0,
            log10sigma=1.5,
            sign=1,
        )
    assert len(comp.get_indices()) == 3

    comp.remove_chrom_gauss_event([1, 3])
    assert len(comp.get_indices()) == 1
    assert 2 in comp.get_indices()


def test_chromgauss_add_force(base_model):
    """Adding an event with force=True should replace the existing one."""
    m = _add_chromgauss_component(base_model)
    comp = m.components["ChromaticGaussianEvent"]

    comp.add_chromatic_gaussian_event(
        epoch=55500.0, log10amp=-6.0, chromidx=2.0, log10sigma=1.5, sign=1, index=1
    )
    assert comp.CHROMGAUSS_LOGAMP_1.value == -6.0

    # Force re-add with a different amplitude
    comp.add_chromatic_gaussian_event(
        epoch=55500.0,
        log10amp=-7.0,
        chromidx=2.0,
        log10sigma=1.5,
        sign=1,
        index=1,
        force=True,
    )
    assert comp.CHROMGAUSS_LOGAMP_1.value == -7.0
    assert len(comp.get_indices()) == 1


def test_chromgauss_add_duplicate_raises(base_model):
    """Adding to an existing index without force should raise ValueError."""
    m = _add_chromgauss_component(base_model)
    comp = m.components["ChromaticGaussianEvent"]

    comp.add_chromatic_gaussian_event(
        epoch=55500.0, log10amp=-6.0, chromidx=2.0, log10sigma=1.5, sign=1, index=1
    )

    with pytest.raises(ValueError, match="already in use"):
        comp.add_chromatic_gaussian_event(
            epoch=55500.0,
            log10amp=-7.0,
            chromidx=2.0,
            log10sigma=1.5,
            sign=1,
            index=1,
        )


@pytest.fixture
def chromgauss_par_model_and_toas():
    """A model with one chromatic Gaussian event loaded from a par string, plus fake TOAs."""
    par = """
        RAJ            05:00:00                     1
        DECJ           15:00:00                     1
        POSEPOCH       55000
        F0             100                          1
        F1             -1e-15                       1
        PEPOCH         55000
        CHROMGAUSS_FREF   1400
        CHROMGAUSS_EPOCH_1      55500               1
        CHROMGAUSS_LOGAMP_1     -6.0                1
        CHROMGAUSS_CHROMIDX_1   2.0                 1
        CHROMGAUSS_SIGN_1       1.0                 0
        CHROMGAUSS_LOGSIG_1     1.5                 1
        TZRMJD         55000
        TZRSITE        @
        TZRFRQ         1400
        EPHEM          DE440
        CLOCK          TT(BIPM2019)
        UNITS          TDB
    """
    m = get_model(StringIO(par))

    np.random.seed(42)
    freqs = np.linspace(500, 1500, 8) * u.MHz
    t = make_fake_toas_uniform(
        54000,
        58000,
        4000,
        m,
        freq=freqs,
        obs="@",
        add_noise=True,
        multi_freqs_in_epoch=True,
    )

    return m, t


def test_chromgauss_from_par(chromgauss_par_model_and_toas):
    """Verify that a ChromaticGaussianEvent model can be read from a par string."""
    m, t = chromgauss_par_model_and_toas

    assert "ChromaticGaussianEvent" in m.components
    comp = m.components["ChromaticGaussianEvent"]
    assert len(comp.get_indices()) == 1
    assert 1 in comp.get_indices()

    assert comp.CHROMGAUSS_EPOCH_1.value == 55500.0
    assert comp.CHROMGAUSS_LOGAMP_1.value == -6.0
    assert comp.CHROMGAUSS_CHROMIDX_1.value == 2.0
    assert comp.CHROMGAUSS_SIGN_1.value == 1.0
    assert comp.CHROMGAUSS_LOGSIG_1.value == 1.5

    res = Residuals(t, m)
    assert np.all(np.isfinite(res.calc_time_resids().value))


def test_chromgauss_fit(chromgauss_par_model_and_toas):
    """Fit a model with a chromatic Gaussian event to simulated TOAs and check parameter recovery."""
    m, t = chromgauss_par_model_and_toas

    res = Residuals(t, m)
    assert res.reduced_chi2 < 1.5

    ftr = Fitter.auto(t, m)
    ftr.fit_toas(maxiter=10)
    assert ftr.resids.reduced_chi2 < 1.5

    for p in m.free_params:
        # Skip parameters with extremely tiny uncertainties
        if ftr.model[p].uncertainty_value < 1e-15:
            continue
        assert abs(m[p].value - ftr.model[p].value) / ftr.model[p].uncertainty_value < 4


@pytest.fixture
def chromgauss_par_two_events_model_and_toas():
    """A model with two chromatic Gaussian events loaded from a par string."""
    par = """
        RAJ            05:00:00                     1
        DECJ           15:00:00                     1
        POSEPOCH       55000
        F0             100                          1
        F1             -1e-15                       1
        PEPOCH         55000
        CHROMGAUSS_FREF   1400
        CHROMGAUSS_EPOCH_1      55500               1
        CHROMGAUSS_LOGAMP_1     -6.0                1
        CHROMGAUSS_CHROMIDX_1   2.0                 1
        CHROMGAUSS_SIGN_1       1.0                 0
        CHROMGAUSS_LOGSIG_1     1.5                 1
        CHROMGAUSS_EPOCH_2      56500               1
        CHROMGAUSS_LOGAMP_2     -5.5                1
        CHROMGAUSS_CHROMIDX_2   2.0                 1
        CHROMGAUSS_SIGN_2       -1.0                0
        CHROMGAUSS_LOGSIG_2     1.3                 1
        TZRMJD         55000
        TZRSITE        @
        TZRFRQ         1400
        EPHEM          DE440
        CLOCK          TT(BIPM2019)
        UNITS          TDB
    """
    m = get_model(StringIO(par))

    np.random.seed(123)
    freqs = np.linspace(500, 1500, 8) * u.MHz
    t = make_fake_toas_uniform(
        54000,
        58000,
        4000,
        m,
        freq=freqs,
        obs="@",
        add_noise=True,
        multi_freqs_in_epoch=True,
    )

    return m, t


def test_chromgauss_from_par_two_events(chromgauss_par_two_events_model_and_toas):
    """Verify that two ChromaticGaussianEvents can be parsed from a par string."""
    m, t = chromgauss_par_two_events_model_and_toas

    comp = m.components["ChromaticGaussianEvent"]
    assert len(comp.get_indices()) == 2
    assert set(comp.get_indices()) == {1, 2}

    assert comp.CHROMGAUSS_EPOCH_2.value == 56500.0
    assert comp.CHROMGAUSS_LOGAMP_2.value == -5.5
    assert comp.CHROMGAUSS_SIGN_2.value == -1.0

    res = Residuals(t, m)
    assert np.all(np.isfinite(res.calc_time_resids().value))


def test_chromgauss_fit_two_events(chromgauss_par_two_events_model_and_toas):
    """Fit a model with two chromatic Gaussian events."""
    m, t = chromgauss_par_two_events_model_and_toas

    res = Residuals(t, m)
    assert res.reduced_chi2 < 1.5

    ftr = Fitter.auto(t, m)
    ftr.fit_toas(maxiter=10)
    assert ftr.resids.reduced_chi2 < 1.5

    for p in m.free_params:
        # Skip parameters with extremely tiny uncertainties
        if ftr.model[p].uncertainty_value < 1e-15:
            continue
        assert abs(m[p].value - ftr.model[p].value) / ftr.model[p].uncertainty_value < 4
