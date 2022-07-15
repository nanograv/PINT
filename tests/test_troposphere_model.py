#!/usr/bin/env python

import os
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from pinttestdata import datadir, testdir

import pint.models as models
import pint.observatory
import pint.toa as toa
from pint.observatory import get_observatory


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    s = Setup()

    s.MIN_ALT = 5  # the minimum altitude in degrees for testing the delay model

    s.FLOAT_THRESHOLD = 1e-12  #

    # parfile = os.path.join(datadir, "J1744-1134.basic.par")
    # ngc = os.path.join(datadir, "NGC6440E")

    s.toas = toa.get_TOAs(
        os.path.join(datadir, "NGC6440E.tim"), picklefilename=pickle_dir
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        s.model = pint.models.get_model(os.path.join(datadir, "NGC6440E.par"))
        s.modelWithTD = pint.models.get_model(os.path.join(datadir, "NGC6440E.par"))
        s.modelWithTD.CORRECT_TROPOSPHERE.value = True

    s.toasInvalid = toa.get_TOAs(
        os.path.join(datadir, "NGC6440E.tim"), picklefilename=pickle_dir
    )

    for i in range(len(s.toasInvalid.table)):
        # adjust the timing by half a day to make them invalid
        s.toasInvalid.table["mjd"][i] += 0.5 * u.d

    s.testAltitudes = np.arange(s.MIN_ALT, 90, 100) * u.deg
    s.testHeights = np.array([10, 100, 1000, 5000]) * u.m

    s.td = s.modelWithTD.components["TroposphereDelay"]
    return s


def test_altitudes(setup):
    # the troposphere delay should strictly decrease with increasing altitude above horizon
    delays = setup.td.delay_model(setup.testAltitudes, 0, 1000 * u.m, 0)

    for i in range(1, len(delays)):
        assert delays[i] < delays[i - 1]


def test_heights(setup):
    # higher elevation observatories should have less troposphere delay
    heightDelays = []  # list of the delays at each height
    for h in setup.testHeights:
        heightDelays.append(setup.td.delay_model(setup.testAltitudes, 0, h, 0))
    for i in range(1, len(setup.testHeights)):
        print(heightDelays[i], heightDelays[i - 1])
        assert np.all(np.less(heightDelays[i], heightDelays[i - 1]))


def test_model_access(setup):
    # make sure that the model components are linked correctly to the troposphere delay
    assert hasattr(setup.model, "CORRECT_TROPOSPHERE")
    assert "TroposphereDelay" in setup.modelWithTD.components.keys()

    # the model should have the sky coordinates defined
    assert setup.td._get_target_skycoord() is not None


def test_invalid_altitudes(setup):
    assert np.all(
        np.less_equal(
            np.abs(setup.td.troposphere_delay(setup.toasInvalid)),
            setup.FLOAT_THRESHOLD * u.s,
        )
    )


def test_latitude_index(setup):
    # the code relies on finding the neighboring latitudes to any site
    # for atmospheric constants defined at every 15 degrees
    # so I will test the nearest neighbors function

    l1 = setup.td._find_latitude_index(20 * u.deg)
    l2 = setup.td._find_latitude_index(-80 * u.deg)
    l3 = setup.td._find_latitude_index(0 * u.deg)
    l4 = setup.td._find_latitude_index(-90 * u.deg)

    assert setup.td.LAT[l1] <= 20 * u.deg < setup.td.LAT[l1 + 1]
    assert setup.td.LAT[l2] <= 80 * u.deg <= setup.td.LAT[l2 + 1]
    assert setup.td.LAT[l3] <= 0 * u.deg <= setup.td.LAT[l3 + 1]
    assert setup.td.LAT[l4] <= 90 * u.deg <= setup.td.LAT[l4 + 1]
