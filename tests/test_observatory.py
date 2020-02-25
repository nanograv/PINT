#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np
import pytest
from pint.pulsar_mjd import Time

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
            assert site, "Observatory {} did not initilized correctly".format(tobs)

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
        if os.getenv("TEMPO2") is None:
            pytest.skip("TEMPO2 evnironment variable is not set, can't run this test")
        # observatory clock correction path expections.
        fake_obs = TopoObs(
            "Fake1",
            tempo_code="?",
            itoa_code="FK",
            clock_fmt="tempo2",
            clock_file="fake2gps.clk",
            clock_dir="TEMPO2",
            itrf_xyz=[0.00, 0.0, 0.0],
        )
        site = get_observatory(
            "Fake1", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
        )
        try:
            site.clock_corrections(self.test_time)
        except (OSError, IOError) as e:
            assert e.errno == 2
            assert os.path.basename(e.filename) == "fake2gps.clk"

    def test_no_tempo2_but_tempo2_clock_requested(self):
        if os.getenv("TEMPO2") is not None:
            pytest.skip("TEMPO2 evnironment variable is set, can't run this test")
        # observatory clock correction path expections.
        fake_obs = TopoObs(
            "Fake1",
            tempo_code="?",
            itoa_code="FK",
            clock_fmt="tempo2",
            clock_file="fake2gps.clk",
            clock_dir="TEMPO2",
            itrf_xyz=[0.00, 0.0, 0.0],
        )
        site = get_observatory(
            "Fake1", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
        )
        with pytest.raises(RuntimeError):
            site.clock_corrections(self.test_time)

    def test_no_tempo_but_tempo_clock_requested(self):
        if os.getenv("TEMPO") is not None:
            pytest.skip("TEMPO evnironment variable is set, can't run this test")
        # observatory clock correction path expections.
        fake_obs = TopoObs(
            "Fake1",
            tempo_code="?",
            itoa_code="FK",
            clock_fmt="tempo",
            clock_file="fake2gps.clk",
            clock_dir="TEMPO",
            itrf_xyz=[0.00, 0.0, 0.0],
        )
        site = get_observatory(
            "Fake1", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
        )
        with pytest.raises(RuntimeError):
            site.clock_corrections(self.test_time)

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
