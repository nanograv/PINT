import os.path

import astropy.units as u
import pytest
from astropy.time import Time

import pint
from pint import models
from pinttestdata import datadir


@pytest.fixture
def model():
    parfile = os.path.join(datadir, "J1600-3053_test.par")
    return models.get_model(parfile)


def test_change_pepoch(model):
    t0 = Time(56000, scale="tdb", format="mjd")
    epoch_diff = (t0.mjd_long - model.PEPOCH.quantity.mjd_long) * u.day
    F0_at_t0 = model.F0.quantity + model.F1.quantity * epoch_diff.to(u.s)
    model.change_pepoch(t0)
    assert model.F0.quantity == F0_at_t0
