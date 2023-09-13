import os
import pytest

import numpy as np
from pint.pulsar_mjd import Time

from pinttestdata import datadir


class TestPsrMjd:
    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.leap_second_days = ["2016-12-31T12:00:00", "2015-06-30T12:00:00"]
        cls.normal_days = ["2016-12-11T12:00:00", "2015-06-02T12:00:00"]

    def test_leap_day(self):
        one_mjd = Time(self.leap_second_days[0], scale="utc")
        mjds = Time(self.leap_second_days, scale="utc")
        # convert to pulsar mjd
        p_one_mjd = Time(one_mjd, format="pulsar_mjd")
        p_mjds = Time(mjds, format="pulsar_mjd")

        assert np.isscalar(
            p_one_mjd.value
        ), "Pulsar one mjd did not return the right lenght."
        assert np.isclose(
            np.modf(p_one_mjd.value)[0], 0.5, atol=1e-14
        ), "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert len(p_mjds) == 2, "Pulsar mjds did not return the right lenght."
        assert np.all(
            np.isclose(np.modf(p_mjds.value)[0], 0.5, atol=1e-14)
        ), "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert not np.all(
            np.isclose(np.modf(mjds.mjd)[0], 0.5, atol=1e-14)
        ), "Astropy time did not have correct leapsecond setup."

    def test_normal_day(self):
        one_mjd = Time(self.normal_days[0], scale="utc")
        mjds = Time(self.normal_days, scale="utc")
        # convert to pulsar mjd
        p_one_mjd = Time(one_mjd, format="pulsar_mjd")
        p_mjds = Time(mjds, format="pulsar_mjd")

        assert np.isscalar(
            p_one_mjd.value
        ), "Pulsar one mjd did not return the right lenght."
        assert np.isclose(
            np.modf(p_one_mjd.value)[0], 0.5, atol=1e-14
        ), "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert len(p_mjds) == 2, "Pulsar mjds did not return the right lenght."
        assert np.all(
            np.isclose(np.modf(p_mjds.value)[0], 0.5, atol=1e-14)
        ), "Pulsar mjd did not give the right fractional day at leapsecond day"
        assert np.all(
            np.isclose(np.modf(mjds.mjd)[0], 0.5, atol=1e-14)
        ), "Astropy time did not have correct leapsecond setup."
