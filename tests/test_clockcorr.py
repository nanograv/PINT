import pytest
from os import path
import io
import numpy as np
from copy import deepcopy

import astropy.units as u
import numpy
import pytest
from astropy.time import Time
from pinttestdata import datadir

from pint.observatory import (
    Observatory,
    get_observatory,
)
from pint.exceptions import ClockCorrectionOutOfRange

from pint.observatory.clock_file import ClockFile
from pint.toa import get_TOAs
from pint.models import get_model_and_toas


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
        with pytest.raises(ClockCorrectionOutOfRange):
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


def test_clk_uncorr():
    m, t = get_model_and_toas(
        datadir / "J0030+0451.mdc1.par", datadir / "J0030+0451.mdc1.tim", allow_tcb=True
    )
    assert m.CLOCK.value == "UNCORR"
    assert not t.clock_corr_info["include_bipm"]
    assert all("clkcorr" not in flags for flags in t.get_flags())


toastr = """
FORMAT 1
unk 0.000000 60000.000000000 0.000 bat 
unk 0.000000 60000.000000000 0.000 gbt 
unk 0.000000 60000.000000000 0.000 coe 
unk 0.000000 60000.000000000 0.000 coe_gps 
"""


def test_bipm_corr():
    # Check that requests for BIPM and GPS corrections are respected
    # And that they are not applied to barycenteric TOAs
    timfile = io.StringIO(toastr)
    obs = get_observatory("coe")
    t = Time(60000.0, scale="utc", format="mjd")
    bipm_delta = obs.bipm_correction(t, bipm_version="BIPM2021")
    gps_delta = obs.gps_correction(t)
    tsYY = get_TOAs(timfile, include_bipm=True, bipm_version="BIPM2021")
    # No correction should have been applied tot the bat TOA
    assert np.abs(tsYY.table["mjd"][0].mjd - t.mjd) < 1.0e-10 / 86400.0
    # COE TOA should have gotten only BIPM correction
    assert np.abs(tsYY.table["mjd"][2] - t - bipm_delta) < 0.1 * u.ns
    # COE_GPS TOA should have gotten both GPS and BIPM correction
    assert np.abs(tsYY.table["mjd"][3] - t - bipm_delta - gps_delta) < 0.1 * u.ns

    # Now check that clock corrections have been backed out before writing to a tim file
    o = io.StringIO()
    tsYY.write_TOA_file(o)
    o.seek(0)
    lines = [ll.strip() for ll in o.readlines()]
    for ll in lines:
        if ll.startswith("C") or ll.startswith("FORMAT"):
            next
        else:
            assert np.abs(float(ll.split()[2]) - 60000.0) < 1.0e-9 / 86400

    # Check that to_TOA_list undoes clock corrections when requested and leaves them in when not
    tsZZ = deepcopy(tsYY)
    lst = tsYY.to_TOA_list(undo_clkcorr=False)
    assert "clkcorr" in lst[1].flags

    lst = tsZZ.to_TOA_list(undo_clkcorr=True)
    assert "clkcorr" not in lst[1].flags

    # Now make sure BIPM corrections are not applied when not requested
    timfile = io.StringIO(toastr)
    tsNN = get_TOAs(timfile, include_bipm=False, bipm_version="BIPM2021")
    # No correction should have been applied to the bat or the COE TOA
    # ACTUALLY the GPS to UTC correction should still have been applied for coe.
    # but at MJD 60000.0 the correction is 0.7 ns so doesn't cause this to fail.
    assert np.abs(tsNN.table["mjd"][0].mjd - t.mjd) < 1.0e-9 / 86400
    assert np.abs(tsNN.table["mjd"][2] - t) < 1.0 * u.ns
