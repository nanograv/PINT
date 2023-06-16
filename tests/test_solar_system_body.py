import os
import pytest

import astropy.time as time
import numpy as np
from astropy.coordinates import solar_system_ephemeris

import pint.config
from pint.solar_system_ephemerides import objPosVel, objPosVel_wrt_SSB
from pinttestdata import datadir


class TestSolarSystemDynamic:
    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        MJDREF = 2400000.5
        J2000_JD = 2451545.0
        J2000_MJD = J2000_JD - MJDREF
        SECPERJULDAY = 86400.0
        ets = np.random.uniform(0.0, 9000.0, 10000) * SECPERJULDAY
        mjd = J2000_MJD + ets / SECPERJULDAY
        cls.tdb_time = time.Time(mjd, scale="tdb", format="mjd")
        cls.ephem = ["de405", "de421", "de434", "de430", "de436"]
        cls.planets = ["jupiter", "saturn", "venus", "uranus", "neptune"]

    # Here we only test if any errors happens.
    def test_earth(self):
        for ep in self.ephem:
            a = objPosVel_wrt_SSB("earth", self.tdb_time, ephem=ep)
            assert a.obj == "earth"
            assert a.pos.shape == (3, 10000)
            assert a.vel.shape == (3, 10000)

    def test_sun(self):
        for ep in self.ephem:
            a = objPosVel_wrt_SSB("sun", self.tdb_time, ephem=ep)
            assert a.obj == "sun"
            assert a.pos.shape == (3, 10000)
            assert a.vel.shape == (3, 10000)

    def test_planets(self):
        for p in self.planets:
            for ep in self.ephem:
                a = objPosVel_wrt_SSB(p, self.tdb_time, ephem=ep)
                assert a.obj == p
                assert a.pos.shape == (3, 10000)
                assert a.vel.shape == (3, 10000)

    def test_earth2obj(self):
        objs = self.planets + ["sun"]
        for obj in objs:
            for ep in self.ephem:
                a = objPosVel("earth", obj, self.tdb_time, ep)
                assert a.obj == obj
                assert a.origin == "earth"
                assert a.pos.shape == (3, 10000)
                assert a.vel.shape == (3, 10000)

    def test_from_dir(self):
        path = pint.config.runtimefile("de432s.bsp")
        a = objPosVel_wrt_SSB("earth", self.tdb_time, "de432s", path=path)
        assert a.obj == "earth"
        assert a.pos.shape == (3, 10000)
        assert a.vel.shape == (3, 10000)
        print("value {0}, path {1}".format(solar_system_ephemeris._value, path))
        # de432s doesn't really exist, does it? so if we got this far it
        # loaded what we told it to
        # assert solar_system_ephemeris._value == path
