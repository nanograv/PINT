import os

import astropy.units as u
from pinttestdata import datadir
import pytest

import pint.fitter
from pint.models import get_model
from pint.toa import get_TOAs


@pytest.mark.parametrize("fitter", [pint.fitter.WLSFitter, pint.fitter.PowellFitter])
def test_fitter_respects_pulse_numbers(fitter):
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_DMX.par"))
    t = get_TOAs(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim"))
    t.compute_pulse_numbers(m)
    F0 = m.F0.quantity.copy()
    m.F0.quantity += 5e-9 * u.Hz
    for p in m.params:
        getattr(m, p).frozen = True
    m.F0.frozen = False
    f = fitter(t, m, track_mode="use_pulse_numbers")
    f.fit_toas()
    assert abs(f.model.F0.quantity - F0) < 1e-10 * u.Hz
