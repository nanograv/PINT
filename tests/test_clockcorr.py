import pytest
from os import path
import io

import astropy.units as u
import numpy
import pytest
from astropy.time import Time
from pinttestdata import datadir

from pint.observatory import Observatory
from pint.observatory.clock_file import ClockFile
from pint.toa import get_TOAs


class TestClockcorrection:
    # Note, these tests currently depend on external data (TEMPO2 clock
    # files, which could potentially change.  Values here are taken
    # from tempo2 version 2020-06-01 or so.
    def test_parkes_agrees_with_tabulated_values(self):
        obs = Observatory.get("Parkes")
        obs.last_clock_correction_mjd()
        cf = obs._clock[0]

        t = Time(51211.73, format="pulsar_mjd", scale="utc")
        assert numpy.isclose(cf.evaluate(t).to(u.us).value, 1.75358956)

        t = Time(55418.11285, format="pulsar_mjd", scale="utc")
        assert numpy.isclose(cf.evaluate(t).to(u.us).value, -0.593622)

        # Test that an error is raised when time is outside of clock file range.
        # Normally it just prints a warning, but I'm not sure how to test for that, so I set limits="error"
        with pytest.raises(RuntimeError):
            t = cf.time[-1] + 1.0 * u.d
            cf.evaluate(t, limits="error")

    def test_wsrt_parsed_correctly_with_text_columns(self):
        # Issue here is the wsrt clock file has text columns, which used to cause np.loadtxt to crash
        cf = ClockFile.read(path.join(datadir, "wsrt2gps.clk"), format="tempo2")

        t = Time(51189.5, scale="utc", format="mjd")
        e = cf.evaluate(t)
        assert numpy.isclose(e.to(u.us).value, 1.093)

        # Test that an error is raised when time is outside of clock file range.
        # Normally it just prints a warning, but I'm not sure how to test for that, so I set limits="error"
        with pytest.raises(RuntimeError):
            t = cf.time[-1] + 1.0 * u.d
            cf.evaluate(t, limits="error")


def test_clockcorr_roundtrip():
    timlines = """FORMAT 1
    toa1 1400 55555.0 1.0 gbt
    toa2 1400 55556.0 1.0 gbt"""
    t = get_TOAs(io.StringIO(timlines))
    # should have positive clock correction applied
    assert t.get_mjds()[0].value > 55555
    assert t.get_mjds()[1].value > 55556
    o = io.StringIO()
    t.write_TOA_file(o)
    o.seek(0)
    lines = o.readlines()
    # make sure the clock corrections are no longer there.
    for line in lines:
        if line.startswith("toa1"):
            assert float(line.split()[2]) == 55555
        if line.startswith("toa2"):
            assert float(line.split()[2]) == 55556
