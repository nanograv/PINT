#!/usr/bin/env python
import os
import unittest

import numpy as np
import pytest
from pint.pulsar_mjd import Time

import pint.observatory
import pint.observatory.observatories
from pint.observatory import get_observatory
from pint.observatory.topo_obs import TopoObs
from pinttestdata import datadir


class TestObservatory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.test_obs = ["aro", "ao", "chime", "drao"]
        cls.test_time = Time(
            np.linspace(55000, 58000, num=100), scale="utc", format="pulsar_mjd"
        )

    def test_get_obs(self):
        for tobs in self.test_obs:
            site = get_observatory(
                tobs, include_gps=False, include_bipm=True, bipm_version="BIPM2015"
            )
            assert site, "Observatory {} did not initialize correctly".format(tobs)

    def test_different_bipm(self):
        for tobs in self.test_obs:
            site = get_observatory(
                tobs, include_gps=False, include_bipm=True, bipm_version="BIPM2019"
            )
            assert site, "BIPM2019 is not a valid BIPM choice"

    def test_clock_corr(self):
        for tobs in self.test_obs:
            site = get_observatory(
                tobs, include_gps=True, include_bipm=True, bipm_version="BIPM2015"
            )
            clock_corr = site.clock_corrections(self.test_time)
            assert len(clock_corr) == len(self.test_time)
            # Test one time
            clock_corr1 = site.clock_corrections(self.test_time[0])
            assert clock_corr1.shape == ()

    def test_get_TDBs(self):
        for tobs in self.test_obs:
            site = get_observatory(
                tobs, include_gps=True, include_bipm=True, bipm_version="BIPM2015"
            )
            # Test default TDB calculation
            tdbs = site.get_TDBs(self.test_time)
            assert len(tdbs) == len(self.test_time)
            tdb1 = site.get_TDBs(self.test_time[0])
            assert tdb1.shape == (1,)

            # Test TDB calculation from ephemeris
            tdbs = site.get_TDBs(self.test_time, method="ephemeris", ephem="de430t")
            assert len(tdbs) == len(self.test_time)
            tdb1 = site.get_TDBs(self.test_time[0], method="ephemeris", ephem="de430t")
            assert tdb1.shape == (1,)

    def test_positions(self):
        for tobs in self.test_obs:
            site = get_observatory(
                tobs, include_gps=True, include_bipm=True, bipm_version="BIPM2015"
            )
            posvel = site.posvel(self.test_time, ephem="de436")
            assert posvel.pos.shape == (3, len(self.test_time))
            assert posvel.vel.shape == (3, len(self.test_time))

    def test_wrong_name(self):
        with pytest.raises(KeyError):
            get_observatory("Wrong_name")

    def test_wrong_TDB_method(self):
        site = get_observatory(
            "ao", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
        )
        with self.assertRaises(ValueError):
            site.get_TDBs(self.test_time, method="ephemeris")
        with self.assertRaises(ValueError):
            site.get_TDBs(self.test_time, method="Unknown_method")
        with self.assertRaises(ValueError):
            site.get_TDBs(self.test_time, method="ephemeris", ephem="de436")


@pytest.mark.parametrize(
    "observatory", list(pint.observatory.Observatory._registry.keys())
)
def test_can_try_to_compute_corrections(observatory):
    # Many of these should emit warnings
    get_observatory(observatory).clock_corrections(Time(57600, format="mjd"))


good_observatories = ["gbt", "ao", "vla", "jodrell", "wsrt", "parkes"]


@pytest.mark.parametrize("observatory", good_observatories)
def test_can_compute_corrections(observatory):
    get_observatory(observatory).clock_corrections(
        Time(55600, format="mjd"), limits="error"
    )


@pytest.mark.parametrize("observatory", good_observatories)
def test_last_mjd(observatory):
    assert get_observatory(observatory).last_clock_correction_mjd() > 55600


def test_missing_clock_gives_exception_nonexistent():
    o = TopoObs(
        "arecibo_bogus",
        clock_file="nonexistent.dat",
        itoa_code="W",
        itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
        overwrite=True,
    )

    with pytest.raises(RuntimeError):
        o.clock_corrections(Time(57600, format="mjd"), limits="error")


def test_missing_clock_gives_exception_no_data():
    o = TopoObs(
        "arecibo_bogus",
        itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
        overwrite=True,
    )

    with pytest.raises(RuntimeError):
        o.clock_corrections(Time(57600, format="mjd"), limits="error")


def test_missing_clock_runs():
    o = TopoObs(
        "arecibo_bogus",
        clock_file="nonexistent.dat",
        itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
        overwrite=True,
    )
    o.clock_corrections(Time(57600, format="mjd"))


def test_observatories_registered():
    assert len(pint.observatory.Observatory._registry) > 5


def test_gbt_registered():
    get_observatory("gbt")


# All these cases are partially tested in tests/test_find_clock_file.py
# In each case you come back with a zero-length clock file, which will
# give warnings or errrors based on limits="error"
# FIXME: add test for what happens when a clock correction file is not available
# FIXME: add test for what happens when a clock correction is supposed to be in $TEMPO2 but $TEMPO2 is not set
# FIXME: ditto $TEMPO
