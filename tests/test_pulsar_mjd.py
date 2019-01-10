#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import unittest
import numpy as np
import os
from astropy.time import Time
from pint import pulsar_mjd
from pinttestdata import testdir, datadir


os.chdir(datadir)


class TestPsrMjd(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.leap_second_days = ["2016-12-31T12:00:00", "2015-06-30T12:00:00"]
        self.normal_days = ["2016-12-11T12:00:00", "2015-06-02T12:00:00"]

    def test_leap_day(self):
        one_mjd = Time(self.leap_second_days[0], scale='utc')
        mjds = Time(self.leap_second_days, scale='utc')
        # convert to pulsar mjd
        p_one_mjd = Time(one_mjd, format='pulsar_mjd')
        p_mjds = Time(mjds, format='pulsar_mjd')

        assert np.isscalar(p_one_mjd.value), \
            "Pulsar one mjd did not return the right lenght."
        assert np.isclose(np.modf(p_one_mjd.value)[0], 0.5, atol=1e-14), \
            "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert len(p_mjds) == 2, \
            "Pulsar mjds did not return the right lenght."
        assert np.all(np.isclose(np.modf(p_mjds.value)[0], 0.5, atol=1e-14)),\
            "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert not np.all(np.isclose(np.modf(mjds.mjd)[0], 0.5, atol=1e-14)), \
            "Astropy time did not have correct leapsecond setup."

    def test_normal_day(self):
        one_mjd = Time(self.normal_days[0], scale='utc')
        mjds = Time(self.normal_days, scale='utc')
        # convert to pulsar mjd
        p_one_mjd = Time(one_mjd, format='pulsar_mjd')
        p_mjds = Time(mjds, format='pulsar_mjd')

        assert np.isscalar(p_one_mjd.value), \
            "Pulsar one mjd did not return the right lenght."
        assert np.isclose(np.modf(p_one_mjd.value)[0], 0.5, atol=1e-14), \
            "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert len(p_mjds) == 2, \
            "Pulsar mjds did not return the right lenght."
        assert np.all(np.isclose(np.modf(p_mjds.value)[0], 0.5, atol=1e-14)),\
            "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert np.all(np.isclose(np.modf(mjds.mjd)[0], 0.5, atol=1e-14)), \
            "Astropy time did not have correct leapsecond setup."
