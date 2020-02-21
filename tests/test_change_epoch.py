import os.path

import astropy.units as u
import numpy as np
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


def test_change_posepoch(model):
    t0 = Time(56000, scale="tdb", format="mjd")
    epoch_diff = (t0.mjd_long - model.POSEPOCH.quantity.mjd_long) * u.day

    orig_coords = model.get_psr_coords()
    lon = orig_coords.spherical.lon
    lat = orig_coords.spherical.lat
    differentials = orig_coords.sphericalcoslat.differentials["s"]
    pm_lon_coslat = differentials.d_lon_coslat
    pm_lat = differentials.d_lat
    new_lon = lon + pm_lon_coslat / np.cos(lat) * epoch_diff
    new_lat = lat + pm_lat * epoch_diff

    model.change_posepoch(t0)
    new_coords = model.get_psr_coords()

    assert np.abs(new_coords.spherical.lon - new_lon) < 40 * u.uas
    assert np.abs(new_coords.spherical.lat - new_lat) < 40 * u.uas
