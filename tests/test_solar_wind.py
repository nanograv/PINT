""" Test for pint solar wind module
"""

import os
from io import StringIO
import pytest
import numpy as np
from numpy.testing import assert_allclose
import copy
import sys

import astropy.units as u
from astropy.time import Time
from pint.models import get_model
from pint.fitter import WidebandTOAFitter
from pint.toa import get_TOAs, make_fake_toas
from pinttestdata import datadir

os.chdir(datadir)

par = """
PSR J1234+5678
F0 1
DM 10
ELAT 0
ELONG 0
PEPOCH 54000
"""

march_equinox = Time("2015-03-20 22:45:00").mjd
year = 365.25  # in mjd


@pytest.mark.parametrize("frac", [0, 0.25, 0.5, 0.123])
def test_sun_angle_ecliptic(frac):
    model = get_model(StringIO(par))
    toas = make_fake_toas(
        march_equinox, march_equinox + 2 * year, 10, model=model, obs="gbt"
    )
    # Sun longitude, from Astronomical Almanac
    sun_n = toas.get_mjds().value - 51544.5
    sun_L = 280.460 + 0.9856474 * sun_n
    sun_g = 357.528 + 0.9856003 * sun_n
    sun_longitude = (
        sun_L
        + 1.915 * np.sin(np.deg2rad(sun_g))
        + 0.020 * np.sin(2 * np.deg2rad(sun_g))
    )
    sun_longitude = (sun_longitude + 180) % 360 - 180
    angles = np.rad2deg(model.sun_angle(toas).value)
    assert_allclose(angles, np.abs(sun_longitude), atol=1)


def test_solar_wind_delays_positive():
    model = get_model(StringIO("\n".join([par, "NE_SW 1"])))
    toas = make_fake_toas(54000, 54000 + year, 13, model=model, obs="gbt")
    assert np.all(model.components["SolarWindDispersion"].solar_wind_dm(toas) > 0)
