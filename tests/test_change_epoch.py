import os.path

import astropy.units as u
from astropy.time import Time

import pint
from pint import models
from pinttestdata import datadir


def test_change_pepoch():
    model = models.get_model(os.path.join(datadir, "J1713+0747_NANOGrav_11yv0.gls.par"))
    t0 = Time(56000, scale="tdb", format="mjd")
    epoch_diff = (t0.mjd_long - model.PEPOCH.quantity.mjd_long) * u.day
    F0_at_t0 = model.F0.quantity + model.F1.quantity * epoch_diff.to(u.s)
    model.change_pepoch(t0)
    assert model.F0.quantity == F0_at_t0
