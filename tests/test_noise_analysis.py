import pytest
from pint.models import get_model
from pint.models.cmwavex import cmwavex_setup
from pint.models.dmwavex import dmwavex_setup
from pint.models.wavex import wavex_setup
from pint.simulation import make_fake_toas_uniform
from pint.fitter import WLSFitter
from pint.utils import find_optimal_nharms
from pint.noise_analysis import (
    compute_aic,
    plchromnoise_from_cmwavex,
    plrednoise_from_wavex,
    pldmnoise_from_dmwavex,
    prepare_model,
)

from io import StringIO
import numpy as np
import astropy.units as u
from copy import deepcopy


@pytest.fixture
def data_wx():
    par_sim_wx = """
            PSR           SIM3
            RAJ           05:00:00     1
            DECJ          15:00:00     1
            PEPOCH        55000
            F0            100          1
            F1            -1e-15       1 
            PHOFF         0            1
            DM            15           1
            TNREDAMP      -12.5
            TNREDGAM      3.5
            TNREDC        10
            TZRMJD        55000
            TZRFRQ        1400 
            TZRSITE       gbt
            UNITS         TDB
            EPHEM         DE440
            CLOCK         TT(BIPM2019)
        """
    m = get_model(StringIO(par_sim_wx))

    ntoas = 200
    toaerrs = np.random.uniform(0.5, 2.0, ntoas) * u.us
    freqs = np.linspace(500, 1500, 2) * u.MHz

    t = make_fake_toas_uniform(
        startMJD=54001,
        endMJD=56001,
        ntoas=ntoas,
        model=m,
        freq=freqs,
        obs="gbt",
        error=toaerrs,
        add_noise=True,
        add_correlated_noise=True,
        name="fake",
        include_bipm=True,
        multi_freqs_in_epoch=True,
    )

    return m, t


@pytest.fixture
def data_dmwx():
    par_sim_dmwx = """
            PSR           SIM3
            RAJ           05:00:00     1
            DECJ          15:00:00     1
            PEPOCH        55000
            F0            100          1
            F1            -1e-15       1 
            PHOFF         0            1
            DM            15           1
            TNDMAMP       -13
            TNDMGAM       3.5
            TNDMC         10
            TZRMJD        55000
            TZRFRQ        1400 
            TZRSITE       gbt
            UNITS         TDB
            EPHEM         DE440
            CLOCK         TT(BIPM2019)
        """

    m = get_model(StringIO(par_sim_dmwx))

    ntoas = 200
    toaerrs = np.random.uniform(0.5, 2.0, ntoas) * u.us
    freqs = np.linspace(500, 1500, 4) * u.MHz

    t = make_fake_toas_uniform(
        startMJD=54001,
        endMJD=56001,
        ntoas=ntoas,
        model=m,
        freq=freqs,
        obs="gbt",
        error=toaerrs,
        add_noise=True,
        add_correlated_noise=True,
        name="fake",
        include_bipm=True,
        multi_freqs_in_epoch=True,
    )

    return m, t


@pytest.fixture
def data_cmwx():
    par_sim_cmwx = """
            PSR           SIM3
            RAJ           05:00:00     1
            DECJ          15:00:00     1
            PEPOCH        55000
            F0            100          1
            F1            -1e-15       1 
            PHOFF         0            1
            DM            15           1
            TNCHROMIDX    4
            CM            10
            TNCHROMAMP    -13
            TNCHROMGAM    3.5
            TNCHROMC      10
            TZRMJD        55000
            TZRFRQ        1400 
            TZRSITE       gbt
            UNITS         TDB
            EPHEM         DE440
            CLOCK         TT(BIPM2019)
        """

    m = get_model(StringIO(par_sim_cmwx))

    ntoas = 200
    toaerrs = np.random.uniform(0.5, 2.0, ntoas) * u.us
    freqs = np.linspace(500, 1500, 4) * u.MHz

    t = make_fake_toas_uniform(
        startMJD=54001,
        endMJD=56001,
        ntoas=ntoas,
        model=m,
        freq=freqs,
        obs="gbt",
        error=toaerrs,
        add_noise=True,
        add_correlated_noise=True,
        name="fake",
        include_bipm=True,
        multi_freqs_in_epoch=True,
    )

    return m, t


def test_wx2pl(data_wx):
    m, t = data_wx

    m1 = deepcopy(m)
    m1.remove_component("PLRedNoise")

    Tspan = t.get_mjds().max() - t.get_mjds().min()
    wavex_setup(m1, Tspan, n_freqs=int(m.TNREDC.value), freeze_params=False)

    ftr = WLSFitter(t, m1)
    ftr.fit_toas(maxiter=10)
    m1 = ftr.model

    m2 = plrednoise_from_wavex(m1)

    assert "PLRedNoise" in m2.components
    assert abs(m.TNREDAMP.value - m2.TNREDAMP.value) / m2.TNREDAMP.uncertainty_value < 5
    assert abs(m.TNREDGAM.value - m2.TNREDGAM.value) / m2.TNREDGAM.uncertainty_value < 5


def test_dmwx2pldm(data_dmwx):
    m, t = data_dmwx

    m1 = deepcopy(m)
    m1.remove_component("PLDMNoise")

    Tspan = t.get_mjds().max() - t.get_mjds().min()
    dmwavex_setup(m1, Tspan, n_freqs=int(m.TNDMC.value), freeze_params=False)

    ftr = WLSFitter(t, m1)
    ftr.fit_toas(maxiter=10)
    m1 = ftr.model

    m2 = pldmnoise_from_dmwavex(m1)

    assert "PLDMNoise" in m2.components
    assert abs(m.TNDMAMP.value - m2.TNDMAMP.value) / m2.TNDMAMP.uncertainty_value < 5
    assert abs(m.TNDMGAM.value - m2.TNDMGAM.value) / m2.TNDMGAM.uncertainty_value < 5


def test_cmwx2pldm(data_cmwx):
    m, t = data_cmwx

    m1 = deepcopy(m)
    m1.remove_component("PLChromNoise")

    Tspan = t.get_mjds().max() - t.get_mjds().min()
    cmwavex_setup(m1, Tspan, n_freqs=int(m.TNCHROMC.value), freeze_params=False)

    ftr = WLSFitter(t, m1)
    ftr.fit_toas(maxiter=10)
    m1 = ftr.model

    m2 = plchromnoise_from_cmwavex(m1)

    assert "PLChromNoise" in m2.components
    assert (
        abs(m.TNCHROMAMP.value - m2.TNCHROMAMP.value) / m2.TNCHROMAMP.uncertainty_value
        < 5
    )
    assert (
        abs(m.TNCHROMGAM.value - m2.TNCHROMGAM.value) / m2.TNCHROMGAM.uncertainty_value
        < 5
    )


def test_find_optimal_nharms_wx(data_wx):
    m, t = data_wx

    m1 = deepcopy(m)
    m1.remove_component("PLRedNoise")

    nharm, aics = find_optimal_nharms(m1, t, "WaveX", nharms_max=7)

    assert nharm <= 7
    assert np.all(aics >= 0)


def test_find_optimal_nharms_dmwx(data_dmwx):
    m, t = data_dmwx

    m1 = deepcopy(m)
    m1.remove_component("PLDMNoise")

    nharm, aics = find_optimal_nharms(m1, t, "DMWaveX", nharms_max=7)

    assert nharm <= 7
    assert np.all(aics >= 0)


@pytest.mark.parametrize(
    "component_names",
    [
        ["WaveX"],
        ["DMWaveX"],
        ["CMWaveX"],
        ["WaveX", "DMWaveX"],
        ["WaveX", "CMWaveX"],
        ["DMWaveX", "CMWaveX"],
        ["WaveX", "DMWaveX", "CMWaveX"],
    ],
)
def test_prepare_model_for_find_optimal_nharms(data_dmwx, component_names):
    m0, t = data_dmwx

    m = prepare_model(
        model=m0,
        Tspan=10 * u.year,
        include_components=component_names,
        nharms=np.repeat(10, len(component_names)),
        chromatic_index=4,
    )

    assert "PLDMNoise" not in m.components

    assert "PHOFF" in m.free_params

    assert ("WaveX" in component_names) == ("WaveX" in m.components)
    assert ("DMWaveX" in component_names) == ("DMWaveX" in m.components)
    assert ("CMWaveX" in component_names) == ("CMWaveX" in m.components)

    if "WaveX" in component_names:
        assert len(m.components["WaveX"].get_indices()) == 10

    if "DMWaveX" in component_names:
        assert not m.DM.frozen and m.DM.quantity is not None
        assert not m.DM1.frozen and m.DM1.quantity is not None
        assert not m.DM2.frozen and m.DM2.quantity is not None
        assert len(m.components["DMWaveX"].get_indices()) == 10

    if "CMWaveX" in component_names:
        assert not m.CM.frozen and m.CM.quantity is not None
        assert not m.CM1.frozen and m.CM1.quantity is not None
        assert not m.CM2.frozen and m.CM2.quantity is not None
        assert len(m.components["CMWaveX"].get_indices()) == 10


@pytest.mark.parametrize(
    "component_names",
    [
        ["WaveX"],
        ["DMWaveX"],
        ["CMWaveX"],
        ["WaveX", "DMWaveX"],
        ["WaveX", "CMWaveX"],
        ["DMWaveX", "CMWaveX"],
        ["WaveX", "DMWaveX", "CMWaveX"],
    ],
)
def test_compute_aic(data_dmwx, component_names):
    m, t = data_dmwx
    assert np.isfinite(
        compute_aic(
            m,
            t,
            include_components=component_names,
            ii=np.array((8, 9, 10)),
            chromatic_index=4,
        )
    )
