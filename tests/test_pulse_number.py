#!/usr/bin/python
from __future__ import absolute_import

import os
import pytest
from copy import deepcopy

import numpy as np
from astropy import units as u

from pint.residuals import Residuals
from pinttestdata import datadir
import pint.fitter
from pint.models import get_model
from pint.toa import get_TOAs, make_fake_toas

parfile = os.path.join(datadir, "withpn.par")
timfile = os.path.join(datadir, "withpn.tim")


def test_pulse_number():
    model = get_model(parfile)
    toas = get_TOAs(timfile)
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


@pytest.mark.parametrize("obs", ["GBT", "AO", "@", "coe"])
def test_make_fake_toas(obs):
    m = get_model(parfile)
    t = make_fake_toas(56000, 59000, 10, m, obs=obs)
    t.table["error"] = 1 * u.us
    r = Residuals(t, m, track_mode="nearest")
    assert np.amax(np.abs(r.phase_resids)) < 1e-6


def test_parameter_overrides_model():
    m = get_model(parfile)
    t = make_fake_toas(56000, 59000, 10, m, obs="@")
    t.table["error"] = 1 * u.us
    t.compute_pulse_numbers(m)
    delta_f = (1 / (t.last_MJD - t.first_MJD)).to(u.Hz)

    m_2 = deepcopy(m)
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


@pytest.mark.parametrize("fitter", [pint.fitter.WLSFitter, pint.fitter.PowellFitter])
def test_fitter_respects_pulse_numbers(fitter):
    m = get_model(parfile)
    t = make_fake_toas(56000, 59000, 10, m, obs="@")
    t.table["error"] = 1 * u.us
    t.compute_pulse_numbers(m)
    delta_f = (1 / (t.last_MJD - t.first_MJD)).to(u.Hz)

    # Unchanged model, fitting should be trivial
    f_0 = fitter(t, m, track_mode="use_pulse_numbers")
    f_0.fit_toas()
    assert abs(f_0.model.F0.quantity - m.F0.quantity) < 0.01 * delta_f

    m_2 = deepcopy(m)
    m_2.F0.quantity += 2 * delta_f
    for p in m_2.params:
        getattr(m_2, p).frozen = True
    m_2.F0.frozen = False

    # Check tracking does the right thing for residuals
    # and that we're wrapping as much as we think we are
    with pytest.raises(ValueError):
        Residuals(t, m_2, track_mode="capybara")
    r = Residuals(t, m_2, track_mode="nearest")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) <= 1
    r = Residuals(t, m_2, track_mode="use_pulse_numbers")
    assert np.amax(r.phase_resids) - np.amin(r.phase_resids) > 1.9

    # Check fitter with and without tracking
    with pytest.raises(ValueError):
        fitter(t, m_2, track_mode="capybara")

    f_1 = fitter(t, m_2, track_mode="nearest")
    f_1.fit_toas()
    assert abs(f_1.model.F0.quantity - m.F0.quantity) > 0.1 * delta_f

    f_2 = fitter(t, m_2, track_mode="use_pulse_numbers")
    f_2.fit_toas()
    assert abs(f_2.model.F0.quantity - m.F0.quantity) < 0.01 * delta_f
