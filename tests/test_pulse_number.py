#!/usr/bin/python
from __future__ import absolute_import

import os
import pytest

import numpy as np
from astropy import units as u

import pint.models
import pint.toa
from pint.residuals import Residuals
from pinttestdata import datadir

parfile = os.path.join(datadir, "withpn.par")
timfile = os.path.join(datadir, "withpn.tim")


def test_pulse_number():
    model = pint.models.get_model(parfile)
    toas = pint.toa.get_TOAs(timfile)
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
