import copy
import io
import logging
import os.path
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from pinttestdata import datadir

import pint
import pint.toa
from pint import models


@pytest.fixture(params=["J1600-3053_test.par", "J2317+1439_ell1h_simple.par"])
def model(request):
    parfile = os.path.join(datadir, request.param)
    return models.get_model(parfile)


@pytest.fixture
def times():
    return pint.toa.get_TOAs_list(
        [pint.toa.TOA(t) for t in np.linspace(56000, 57000, 5)]
    )


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
    assert np.abs(model.DM.quantity - DM_at_t0) < 1e-8 * u.pc / u.cm**3


def test_change_dmepoch_times(times):
    parfile = os.path.join(datadir, "J2229+2643_dm1.par")
    model = models.get_model(parfile)
    dms = model.base_dm(times)
    model.change_dmepoch(Time(56000, scale="tdb", format="mjd"))
    assert np.all(np.abs(model.base_dm(times) - dms) < 1e-8 * u.pc / u.cm**3)


def test_change_dmepoch_unset(times):
    model = models.get_model(
        io.StringIO(
            """
            PSR J1234+5678
            F0 1
            PEPOCH 56000
            ELAT 0
            ELONG 0
            DM 10
            """
        )
    )
    dms = model.base_dm(times)
    model.change_dmepoch(Time(56000, scale="tdb", format="mjd"))
    assert np.all(np.abs(model.base_dm(times) - dms) < 1e-8 * u.pc / u.cm**3)


def test_change_dmepoch_unset_exception(times):
    model = models.get_model(
        io.StringIO(
            """
            PSR J1234+5678
            F0 1
            PEPOCH 56000
            ELAT 0
            ELONG 0
            DM 10
            DM1 1e-10
            """
        )
    )
    assert model.DMEPOCH.quantity == model.PEPOCH.quantity


def test_change_dmepoch_unset_python_exception(times):
    model = models.get_model(
        io.StringIO(
            """
            PSR J1234+5678
            F0 1
            PEPOCH 56000
            ELAT 0
            ELONG 0
            DM 10
            """
        )
    )
    model.DM1.value = 7
    with pytest.raises(ValueError):
        model.change_dmepoch(56000)
    model.validate()
    model.change_dmepoch(56000)
    assert model.DMEPOCH.quantity == model.PEPOCH.quantity


def test_unset_dmepoch_raises(times):
    model = models.get_model(
        io.StringIO(
            """
            PSR J1234+5678
            F0 1
            PEPOCH 56000
            ELAT 0
            ELONG 0
            DM 10
            """
        )
    )
    model.DM1.value = 7
    with pytest.raises(ValueError):
        model.base_dm(times)
    with pytest.raises(ValueError):
        model.d_dm_d_DMs(times, "DM")
    model.validate()
    model.base_dm(times)
    model.d_dm_d_DMs(times, "DM")
    assert model.DMEPOCH.quantity == model.PEPOCH.quantity


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
    print("timing model", model.params)
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
            PBDOT = -model.FB1.quantity / model.FB0.quantity**2
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

    t1 = new_epoch + PB / 3
    model_2 = copy.deepcopy(model)
    model_2.change_binary_epoch(t1)
    for p in model.params:
        assert getattr(model, p).quantity == getattr(model_2, p).quantity


test_par = """
PSR J1234+5678
ELAT 0
ELONG 0
F0 1
PEPOCH 58000
DM 10
"""

test_par_radec = """
PSR J1234+5678
RA 10:23:47.67
DEC 00:38:41.2
F0 1
PEPOCH 58000
DM 10
"""


@pytest.mark.parametrize("par", [test_par, test_par_radec])
def test_unset_other_epochs(caplog, par):
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = models.get_model(io.StringIO(par))
        assert m.PEPOCH.quantity is not None
        assert m.DMEPOCH.quantity is None
        assert m.POSEPOCH.quantity is None
    assert not caplog.records


def test_unset_other_epochs_dm1(caplog):
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = models.get_model(io.StringIO("\n".join([test_par, "DM1 1e-10"])))
        assert m.PEPOCH.quantity is not None
        assert m.DMEPOCH.quantity == m.PEPOCH.quantity
        assert m.POSEPOCH.quantity is None
    assert not caplog.records


def test_unset_other_epochs_pmra(caplog):
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = models.get_model(
            io.StringIO("\n".join([test_par_radec, "PMRA 1e-10", "PMDEC 1e-10"]))
        )
        assert m.PEPOCH.quantity is not None
        assert m.DMEPOCH.quantity is None
        assert m.POSEPOCH.quantity == m.PEPOCH.quantity
    assert not caplog.records


def test_unset_other_epochs_pmelat(caplog):
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = models.get_model(
            io.StringIO("\n".join([test_par, "PMELAT 1e-10", "PMELONG 1e-10"]))
        )
        assert m.PEPOCH.quantity is not None
        assert m.DMEPOCH.quantity is None
        assert m.POSEPOCH.quantity == m.PEPOCH.quantity
    assert not caplog.records
