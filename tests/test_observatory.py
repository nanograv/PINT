#!/usr/bin/env python
import io
import os
import unittest
from pathlib import Path

import numpy as np
import pytest
from pinttestdata import datadir

import pint.observatory
import pint.observatory.topo_obs
from pint.observatory import NoClockCorrections, Observatory, get_observatory
from pint.observatory.special_locations import load_special_locations
from pint.observatory.topo_obs import (
    TopoObs,
    load_observatories,
    load_observatories_from_usual_locations,
)
from pint.pulsar_mjd import Time


@pytest.fixture
def sandbox():
    class Sandbox:
        pass

    o = Sandbox()
    e = os.environ.copy()

    try:
        del os.environ["PINT_OBS_OVERRIDE"]
    except KeyError:
        pass
    reg = pint.observatory.Observatory._registry.copy()
    try:
        yield o
    finally:
        os.environ = e
        pint.observatory.Observatory._registry = reg


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

    def test_clock_correction_file_not_available(self):
        r = Observatory._registry.copy()
        try:
            # observatory clock correction path expections.
            TopoObs(
                "Fake1",
                tempo_code="?",
                itoa_code="FK",
                clock_fmt="tempo2",
                clock_file="fake2gps.clk",
                itrf_xyz=[0.00, 0.0, 0.0],
                overwrite=True,
            )
            site = get_observatory(
                "Fake1", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
            )
            with pytest.raises(NoClockCorrections):
                site.clock_corrections(self.test_time, limits="error")
        finally:
            Observatory._registry = r

    def test_no_tempo2_but_tempo2_clock_requested(self):
        r = Observatory._registry.copy()
        try:
            fake_obs = TopoObs(
                "Fake1",
                tempo_code="?",
                itoa_code="FK",
                clock_fmt="tempo2",
                clock_file="fake2gps.clk",
                clock_dir="TEMPO2",
                itrf_xyz=[0.00, 0.0, 0.0],
                overwrite=True,
            )
            site = get_observatory(
                "Fake1", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
            )
            with pytest.raises(RuntimeError):
                site.clock_corrections(self.test_time, limits="error")
        finally:
            Observatory._registry = r

    def test_no_tempo_but_tempo_clock_requested(self):
        r = Observatory._registry.copy()
        try:
            fake_obs = TopoObs(
                "Fake1",
                tempo_code="?",
                itoa_code="FK",
                clock_fmt="tempo",
                clock_file="fake2gps.clk",
                clock_dir="TEMPO",
                itrf_xyz=[0.00, 0.0, 0.0],
                overwrite=True,
            )
            site = get_observatory(
                "Fake1", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
            )
            with pytest.raises(RuntimeError):
                site.clock_corrections(self.test_time, limits="error")
        finally:
            Observatory._registry = r

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


# Some of these now require TEMPO2 clock files
# good_observatories = ["gbt", "ao", "vla", "jodrell", "wsrt", "parkes"]
good_observatories = ["gbt", "ao", "vla", "jodrell", "jbroach", "jbdfb"]


@pytest.mark.parametrize("observatory", good_observatories)
def test_can_compute_corrections(observatory):
    get_observatory(observatory).clock_corrections(
        Time(55600, format="mjd"), limits="error"
    )


@pytest.mark.parametrize("observatory", good_observatories)
def test_last_mjd(observatory):
    assert get_observatory(observatory).last_clock_correction_mjd() > 55600


def test_missing_clock_gives_exception_nonexistent():
    r = Observatory._registry.copy()
    try:
        o = TopoObs(
            "arecibo_bogus",
            clock_file="nonexistent.dat",
            itoa_code="W",
            itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
            overwrite=True,
        )

        with pytest.raises(RuntimeError):
            o.clock_corrections(Time(57600, format="mjd"), limits="error")
    finally:
        Observatory._registry = r


def test_no_clock_means_no_corrections():
    r = Observatory._registry.copy()
    try:
        o = TopoObs(
            "arecibo_bogus",
            itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
        )
        o.clock_corrections(Time(57600, format="mjd"), limits="error")
    finally:
        Observatory._registry = r


def test_observatories_registered():
    assert len(pint.observatory.Observatory._registry) > 5


def test_gbt_registered():
    get_observatory("gbt")


def test_is_gbt_still_ok():

    gbt = get_observatory("gbt")
    assert gbt._loc_itrf.y < 0


@pytest.mark.parametrize("overwrite", [True, False])
def test_observatory_override(sandbox, overwrite):
    gbt_orig = get_observatory("gbt")
    # just like the original GBT, but ITRF Y is positive here, and negative in the real one
    wronggbt = r"""
    {
        "gbt": {
        "tempo_code": "1",
        "itoa_code": "GB",
        "clock_file": "time_gbt.dat",
        "itrf_xyz": [
            882589.289,
            4924872.368,
            3943729.418
        ],
        "origin": "The Robert C. Byrd Green Bank Telescope.\nThis data was obtained by Joe Swiggum from Ryan Lynch in 2021 September.\n"
        }
    }
    """

    if not overwrite:
        with pytest.raises(ValueError):
            load_observatories(io.StringIO(wronggbt), overwrite=overwrite)
    else:
        load_observatories(io.StringIO(wronggbt), overwrite=overwrite)
        newgbt = get_observatory("gbt")
        assert newgbt._loc_itrf.y > 0
        assert newgbt._loc_itrf.y != gbt_orig._loc_itrf.y


def test_list_last_correction_mjds_runs():
    pint.observatory.list_last_correction_mjds()
