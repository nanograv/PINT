import os.path
from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Latitude, Longitude

from pint.models import get_model
from pinttestdata import datadir


@pytest.fixture
def model():
    parfile = os.path.join(datadir, "J1600-3053_test.par")
    return get_model(parfile)


def test_ecliptic_to_icrs(model):
    astrometry_component = model.components["AstrometryEcliptic"]
    params_icrs = astrometry_component.get_params_as_ICRS()

    # Values from NANOGrav 11-year timing paper (Arzoumanian et al. 2018)
    # Ecliptic coords in test par file are from the same source.
    ref_ra = Longitude("16:00:51.903178", u.hourangle)
    ref_dec = Latitude("-30:53:49.3919", u.deg)

    assert np.abs(params_icrs["RAJ"] - ref_ra) < 40 * u.uas
    assert np.abs(params_icrs["DECJ"] - ref_dec) < 40 * u.uas


par_basic_ecliptic = """
PSR J1234+5678
F0 1
PEPOCH 57000
ELAT 0
ELONG 0
DM 10
"""

par_basic_equatorial = """
PSR J1234+5678
F0 1
PEPOCH 57000
RAJ 10:23:47.67
DECJ 00:38:41.2
DM 10
"""


def test_pm_unset():
    m = get_model(StringIO(par_basic_ecliptic))
    assert m.PMELAT.value == 0
    assert m.PMELONG.value == 0
    assert m.POSEPOCH.quantity is None


def test_pm_unset_equatorial():
    m = get_model(StringIO(par_basic_equatorial))
    assert m.PMRA.value == 0
    assert m.PMDEC.value == 0
    assert m.POSEPOCH.quantity is None


def test_pm_acquires_posepoch():
    m = get_model(StringIO(par_basic_ecliptic))
    assert m.POSEPOCH.quantity is None
    m.PMELAT.value = 7
    m.validate()
    assert m.POSEPOCH.quantity == m.PEPOCH.quantity


def test_pm_acquires_posepoch_equatorial():
    m = get_model(StringIO(par_basic_equatorial))
    assert m.POSEPOCH.quantity is None
    m.PMRA.value = 7
    m.validate()
    assert m.POSEPOCH.quantity == m.PEPOCH.quantity


def test_pm_one_set_other_not():
    m = get_model(StringIO("\n".join([par_basic_equatorial, "PMRA 7"])))
    assert m.POSEPOCH.quantity == m.PEPOCH.quantity
