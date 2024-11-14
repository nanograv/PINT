from io import StringIO
import os
import pytest

import astropy.units as u
import numpy as np

from pint.models import get_model
from pint import toa
import pint.residuals
from pinttestdata import datadir

par_nowave = """
    PSRJ           J0835-4510
    RAJ            08:35:20.61149
    DECJ           -45:10:34.8751
    F0             11.18965156782
    PEPOCH         55305                       
    DM             67.99
    UNITS          TDB
"""

wave_terms = """
    WAVEEPOCH 55305
    WAVE_OM 0.0015182579855022
    WAVE1 -0.21573979911255 -0.049063841960712
    WAVE2 0.62795320246729 -0.11984954655989
    WAVE3 0.099618608456845 0.28530756908788
    WAVE4 -0.21537340649058 0.18849486610628
    WAVE5 0.021980474493165 -0.23566696662127
"""


def test_wave_ephem():
    parfile = os.path.join(datadir, "vela_wave.par")
    timfile = os.path.join(datadir, "vela_wave.tim")
    m = get_model(parfile)
    t = toa.get_TOAs(timfile, ephem="DE405", include_bipm=False)
    rs = pint.residuals.Residuals(t, m).time_resids
    assert rs.std() < 350.0 * u.us


def test_wave_construction():
    m = get_model(StringIO(par_nowave + wave_terms))
    assert np.allclose(m.WAVE_OM.quantity, 0.0015182579855022 * u.rad / u.day)
    assert np.allclose(m.WAVE1.quantity[0], -0.21573979911255 * u.s)
    assert np.allclose(m.WAVE2.quantity[1], -0.1198495465598 * u.s)


def test_wave_computation():
    m0 = get_model(StringIO(par_nowave))
    m1 = get_model(StringIO(par_nowave + wave_terms))
    # make some barycentric TOAs
    tdbs = np.linspace(54500, 60000, 10)
    ts = toa.TOAs(
        toalist=[
            toa.TOA(t, obs="@", freq=np.inf, error=1 * u.ms, scale="tdb") for t in tdbs
        ]
    )
    ts.compute_TDBs(ephem="DE421")
    ts.compute_posvels(ephem="DE421")
    ph0 = m0.phase(ts)
    ph1 = m1.phase(ts)
    dphi = (ph1.int - ph0.int) + (ph1.frac - ph0.frac)
    test_phase = np.zeros(len(tdbs))
    WAVEEPOCH = 55305
    WAVE_OM = 0.0015182579855022
    WAVE = [
        [-0.21573979911255, -0.049063841960712],
        [0.62795320246729, -0.11984954655989],
        [0.099618608456845, 0.28530756908788],
        [-0.21537340649058, 0.18849486610628],
        [0.021980474493165, -0.23566696662127],
    ]
    ph = (tdbs - WAVEEPOCH) * WAVE_OM
    for i in range(5):
        test_phase += WAVE[i][0] * np.sin((i + 1) * ph) + WAVE[i][1] * np.cos(
            (i + 1) * ph
        )
    test_phase *= m0.F0.quantity.to(u.Hz).value
    assert np.allclose(test_phase, dphi)
