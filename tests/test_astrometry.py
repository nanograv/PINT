import os.path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Latitude, Longitude

from pint import models
from pinttestdata import datadir


@pytest.fixture
def model():
    parfile = os.path.join(datadir, "J1600-3053_test.par")
    return models.get_model(parfile)


def test_ecliptic_to_icrs(model):
    astrometry_component = model.components["AstrometryEcliptic"]
    params_icrs = astrometry_component.get_params_as_ICRS()

    # Values from NANOGrav 11-year timing paper (Arzoumanian et al. 2018)
    # Ecliptic coords in test par file are from the same source.
    ref_ra = Longitude("16:00:51.903178", u.hourangle)
    ref_dec = Latitude("-30:53:49.3919", u.deg)

    assert np.abs(params_icrs["RAJ"] - ref_ra) < 40 * u.uas
    assert np.abs(params_icrs["DECJ"] - ref_dec) < 40 * u.uas
