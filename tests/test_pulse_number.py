import contextlib
from io import StringIO
import os
import pytest
from copy import deepcopy

import numpy as np
from astropy import units as u
from numpy.testing import assert_almost_equal

from pint.residuals import Residuals
from pinttestdata import datadir
import pint.fitter
from pint.models import get_model
from pint.toa import get_TOAs
from pint.simulation import make_fake_toas_uniform

parfile = os.path.join(datadir, "withpn.par")
timfile = os.path.join(datadir, "withpn.tim")


@pytest.fixture
def model():
    return get_model(parfile)


@pytest.fixture
def toas():
    # The scope="module" setting ensures the TOAs object will be created
    # only once for the whole module, which will save time but might
    # allow accidental modifications done in one test to affect other tests.
    return get_TOAs(timfile)


@pytest.fixture
def fake_toas(model):
    t = make_fake_toas_uniform(56000, 59000, 10, model, obs="@")
    t.table["error"] = 1 * u.us
    return t


def test_pulse_number(model, toas):
    # Make sure pn table column was added
    assert "pulse_number" in toas.table.colnames

    # Tracking pn should result in runaway residuals
    track_resids = Residuals(toas, model).time_resids
    assert np.amax(track_resids) >= 0.2 * u.second

    # Not tracking pn should keep residuals bounded
    getattr(model, "TRACK").value = "0"
    notrack_resids = Residuals(toas, model).time_resids
    assert np.amax(notrack_resids) < 0.2 * u.second

    # Make sure Exceptions are thrown when trying to track nonexistent pn
    del toas.table["pulse_number"]
    getattr(model, "TRACK").value = "-2"
    with pytest.raises(Exception):
        Residuals(toas, model)

    # Make sure pn can be added back by using the model
    assert toas.get_pulse_numbers() is None
    toas.compute_pulse_numbers(model)
    assert "pulse_number" in toas.table.colnames


def test_remove_pulse_number(model, toas):
    assert "pulse_number" in toas.table.colnames
    toas.remove_pulse_numbers()
    assert "pulse_number" not in toas.table.colnames


@pytest.mark.parametrize("obs", ["GBT", "AO", "@", "coe"])
def test_make_fake_toas(obs, model):
    t = make_fake_toas_uniform(56000, 59000, 10, model, obs=obs)
    t.table["error"] = 1 * u.us
    r = Residuals(t, model, track_mode="nearest")
    assert np.amax(np.abs(r.phase_resids)) < 1e-6


def test_parameter_overrides_model(model):
    t = make_fake_toas_uniform(56000, 59000, 10, model, obs="@")
    t.table["error"] = 1 * u.us
    delta_f = (1 / (t.last_MJD - t.first_MJD)).to(u.Hz)

    m_2 = deepcopy(model)
    m_2.F0.quantity += 2 * delta_f

    m_2.TRACK.value = "-2"

    r = Residuals(t, m_2)
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) > 1
    r = Residuals(t, m_2, track_mode="nearest")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) <= 1
    r = Residuals(t, m_2, track_mode="use_pulse_numbers")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) > 1

    m_2.TRACK.value = "0"

    r = Residuals(t, m_2)
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) <= 1
    r = Residuals(t, m_2, track_mode="nearest")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) <= 1
    r = Residuals(t, m_2, track_mode="use_pulse_numbers")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) > 1


def test_residual_respects_pulse_numbers(model, fake_toas):
    t = fake_toas
    delta_f = (1 / (t.last_MJD - t.first_MJD)).to(u.Hz)
    m_2 = deepcopy(model)
    m_2.F0.quantity += 2 * delta_f

    # Check tracking does the right thing for residuals
    # and that we're wrapping as much as we think we are
    with pytest.raises(ValueError):
        Residuals(t, m_2, track_mode="capybara")
    r = Residuals(t, m_2, track_mode="nearest")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) <= 1
    r = Residuals(t, m_2, track_mode="use_pulse_numbers")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) > 1.9


@pytest.mark.parametrize(
    "fitter",
    [
        pint.fitter.WLSFitter,
        pint.fitter.GLSFitter,
        pint.fitter.DownhillWLSFitter,
        pint.fitter.DownhillGLSFitter,
    ],
)
def test_fitter_respects_pulse_numbers(fitter, model, fake_toas):
    t = fake_toas
    delta_f = (1 / (t.last_MJD - t.first_MJD)).to(u.Hz)

    # Unchanged model, fitting should be trivial
    f_0 = fitter(t, model, track_mode="use_pulse_numbers")
    f_0.fit_toas()
    assert abs(f_0.model.F0.quantity - model.F0.quantity) < 0.01 * delta_f

    m_2 = deepcopy(model)
    m_2.F0.quantity += 2 * delta_f
    m_2.free_params = ["F0"]

    # Check fitter with and without tracking
    with pytest.raises(ValueError):
        fitter(t, m_2, track_mode="capybara")

    f_1 = fitter(t, m_2, track_mode="nearest")
    with contextlib.suppress(ValueError):
        f_1.fit_toas()
        assert abs(f_1.model.F0.quantity - model.F0.quantity) > 0.1 * delta_f
    f_2 = fitter(t, m_2, track_mode="use_pulse_numbers")
    f_2.fit_toas()
    assert abs(f_2.model.F0.quantity - model.F0.quantity) < 0.01 * delta_f


@pytest.mark.xfail(reason="partial pulse numbers not supported yet")
def test_partial_pulse_numbers(model, toas):
    r = Residuals(toas, model, track_mode="use_pulse_numbers")
    toas_2 = deepcopy(toas)
    toas_2.table["pulse_number"][1] = np.nan
    r2 = Residuals(toas_2, model, track_mode="use_pulse_numbers")
    assert_almost_equal(r.time_resids[2:].value, r2.time_resids[2:].value)


def test_save_delta_pulse_number(model, toas):
    toas.compute_pulse_numbers(model)
    toas["delta_pulse_number"][:10] = 1
    toas["delta_pulse_number"][10:20] = -1
    outtim = StringIO()
    toas.write_TOA_file(outtim)
    outtim.seek(0)
    newtoas = get_TOAs(outtim)
    assert (newtoas[:10]["delta_pulse_number"] == 1).all()
    assert (newtoas[10:20]["delta_pulse_number"] == -1).all()
    assert (newtoas[20:]["delta_pulse_number"] == 0).all()
    assert (toas["delta_pulse_number"] == newtoas["delta_pulse_number"]).all()


def test_get_pulse_numbers(model, toas):
    toas.compute_pulse_numbers(model)
    pn_column = toas.get_pulse_numbers()

    toas["pn"] = pn_column
    # this should raise an error since both the column and flag are present
    with pytest.raises(ValueError):
        pn_flag = toas.get_pulse_numbers()

    del toas.table["pulse_number"]
    # now it should be OK
    pn_flag = toas.get_pulse_numbers()
    assert (pn_column == pn_flag).all()
