import os.path

import astropy.units as u
import numpy as np
import pint
import pytest
from astropy.time import Time
from pint import models

from pinttestdata import datadir


@pytest.fixture(params=["J1600-3053_test.par", "J2317+1439_ell1h_simple.par"])
def model(request):
    parfile = os.path.join(datadir, request.param)
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


def test_change_dmepoch():
    parfile = os.path.join(datadir, "J2229+2643_dm1.par")
    model = models.get_model(parfile)
    t0 = Time(56000, scale="tdb", format="mjd")
    epoch_diff = (t0.mjd_long - model.DMEPOCH.quantity.mjd_long) * u.day
    DM_at_t0 = model.DM.quantity + model.DM1.quantity * epoch_diff.to(u.s)
    model.change_dmepoch(t0)
    assert np.abs(model.DM.quantity - DM_at_t0) < 1e-8 * u.pc / u.cm ** 3


@pytest.fixture(
    params=[
        "J1737+0811_bt_simple.par",
        "J0023+0923_ell1_simple.par",
        "J2317+1439_ell1h_simple.par",
        "J1955+2908_dd_simple.par",
        "J1713+0747_ddk_simple.par",
        "J0437-4715.par",
    ]
)
def binary_model(request):
    parfile = os.path.join(datadir, request.param)
    return models.get_model(parfile)


def test_change_binary_epoch(binary_model):
    model = binary_model
    t0 = Time(56000, scale="tdb", format="mjd")

    model_kind = model.binary_model_name
    epoch_name = "TASC" if model_kind in ["ELL1", "ELL1H"] else "T0"
    orig_epoch = getattr(model, epoch_name).quantity

    # Get PB and PBDOT from model
    if model.PB.quantity is not None:
        PB = model.PB.quantity
        if model.PBDOT.quantity is not None:
            PBDOT = model.PBDOT.quantity
        else:
            PBDOT = 0.0 * u.Unit("")
    else:
        PB = 1.0 / model.FB0.quantity
        try:
            PBDOT = -model.FB1.quantity / model.FB0.quantity ** 2
        except AttributeError:
            PBDOT = 0.0 * u.Unit("")

    model.change_binary_epoch(t0)
    new_epoch = getattr(model, epoch_name).quantity
    elapsed_time = (new_epoch.mjd_long - orig_epoch.mjd_long) * u.day
    elapsed_periods = elapsed_time / (PB + PBDOT * elapsed_time / 2)
    elapsed_periods = elapsed_periods.to(u.Unit(""))
    n = np.round(elapsed_periods)

    # new_epoch is very close to an integer number of binary periods
    # away from orig_epoch
    assert np.abs(elapsed_periods - n) < 1e-12

    start_time = new_epoch - PB / 2 - (n - 0.5) / 2 * PB * PBDOT
    end_time = new_epoch + PB / 2 + (n + 0.5) / 2 * PB * PBDOT

    # new_epoch is within a single binary period of t0
    assert start_time < t0 < end_time
