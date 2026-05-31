import pytest
import os
import pytest

from astropy import units as u
import pint.models
import pint.toa
import pint.residuals
import pint.simulation
from pinttestdata import datadir

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "zerophase.tim")


class TestAbsPhase:
    def test_phase_zero(self):
        # Check that model phase is 0.0 for a TOA at exactly the TZRMJD
        model = pint.models.get_model(parfile)
        toas = pint.toa.get_TOAs(timfile)

        ph = model.phase(toas, abs_phase=True)
        # Check that integer and fractional phase values are very close to 0.0
        assert ph.int.value == pytest.approx(0.0)
        assert ph.frac.value == pytest.approx(0.0)


def test_tzr_attr():
    model = pint.models.get_model(parfile)
    toas = pint.toa.get_TOAs(timfile)

    assert not toas.tzr
    assert model.components["AbsPhase"].get_TZR_toa(toas).tzr


def test_zero_TZR():
    model = pint.models.get_model(parfile)
    toas = pint.simulation.make_fake_toas_uniform(50000, 51000, 20, model=model)
    toas.adjust_TOAs(10 * u.s)
    r = pint.residuals.Residuals(toas, model)
    assert r.calc_time_mean() > 9 * u.s
    r.zero_TZR()
    r.update()
    assert r.calc_time_mean() < 1 * u.ms
