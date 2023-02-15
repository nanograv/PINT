import numpy as np
from astropy import units as u, constants as c
from astropy.time import Time
import os
from pinttestdata import datadir
import pytest

from pint.models import get_model


def test_fb():
    # with FB terms
    m = get_model(os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par"))
    assert np.isclose(m.pb()[0].to_value(u.d), (1 / m.FB0.quantity).to_value(u.d))


@pytest.mark.parametrize(
    "t",
    [
        Time(55555, format="pulsar_mjd", scale="tdb", precision=9),
        55555 * u.d,
        55555.0,
        55555,
        "55555",
        np.array([55555, 55556]),
    ],
)
def test_fb_input(t):
    # with FB terms
    m = get_model(os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par"))
    pb, pberr = m.pb(t)


def test_pb():
    m = get_model(os.path.join(datadir, "J0437-4715.par"))
    assert np.isclose(m.pb()[0].to_value(u.d), m.PB.quantity.to_value(u.d))


@pytest.mark.parametrize(
    "t",
    [
        Time(55555, format="pulsar_mjd", scale="tdb", precision=9),
        55555 * u.d,
        55555.0,
        55555,
        "55555",
        np.array([55555, 55556]),
    ],
)
def test_pb(t):
    m = get_model(os.path.join(datadir, "J0437-4715.par"))
    pb, pberr = m.pb(t)
