from pint import DMconst
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.fitter import WLSFitter
from pint.utils import (
    dmwavex_setup,
    wavex_setup,
    plrednoise_from_wavex,
    pldmnoise_from_dmwavex,
)

from io import StringIO
import numpy as np
import astropy.units as u
from copy import deepcopy


def test_wx2pl():
    par_sim = """
        PSR           SIM3
        RAJ           05:00:00     1
        DECJ          15:00:00     1
        PEPOCH        55000
        F0            100          1
        F1            -1e-15       1 
        PHOFF         0            1
        DM            15           1
        TNREDAMP      -13
        TNREDGAM      3.5
        TNREDC        5
        TZRMJD        55000
        TZRFRQ        1400 
        TZRSITE       gbt
        UNITS         TDB
        EPHEM         DE440
        CLOCK         TT(BIPM2019)
    """

    m = get_model(StringIO(par_sim))

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
        include_gps=True,
        multi_freqs_in_epoch=True,
    )

    m1 = deepcopy(m)
    m1.remove_component("PLRedNoise")

    Tspan = t.get_mjds().max() - t.get_mjds().min()
    wavex_setup(m1, Tspan, n_freqs=int(m.TNREDC.value))

    for p in m1.params:
        if p.startswith("WXSIN") or p.startswith("WXCOS"):
            m1[p].frozen = False

    ftr = WLSFitter(t, m1)
    ftr.fit_toas(maxiter=3)
    m1 = ftr.model

    m2 = plrednoise_from_wavex(m1)

    assert "PLRedNoise" in m2.components


def test_dmwx2pldm():
    par_sim = """
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
        TNDMC         5
        TZRMJD        55000
        TZRFRQ        1400 
        TZRSITE       gbt
        UNITS         TDB
        EPHEM         DE440
        CLOCK         TT(BIPM2019)
    """

    m = get_model(StringIO(par_sim))

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
        include_gps=True,
        multi_freqs_in_epoch=True,
    )

    m1 = deepcopy(m)
    m1.remove_component("PLDMNoise")

    Tspan = t.get_mjds().max() - t.get_mjds().min()
    dmwavex_setup(m1, Tspan, n_freqs=int(m.TNDMC.value))

    for p in m1.params:
        if p.startswith("DMWXSIN") or p.startswith("DMWXCOS"):
            m1[p].frozen = False

    ftr = WLSFitter(t, m1)
    ftr.fit_toas(maxiter=3)
    m1 = ftr.model

    m2 = pldmnoise_from_dmwavex(m1)

    assert "PLDMNoise" in m2.components
