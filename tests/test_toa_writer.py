import os
import os.path
import pytest

from pint import toa
from pinttestdata import datadir
from astropy.time import Time
import numpy as np
import astropy.units as u


def test_roundtrip_bary_toa_Tempo2format(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="tdb")
    t1 = toa.TOA(t1time, obs="Barycenter", freq=0.0)
    ts = toa.get_TOAs_list([t1], ephem="DE421")
    outnm = os.path.join(tmpdir, "testbaryT.tim")
    ts.write_TOA_file(outnm, format="Tempo2")
    ts2 = toa.get_TOAs(outnm)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_roundtrip_bary_toa_TEMPOformat(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="tdb")
    t1 = toa.TOA(t1time, obs="Barycenter", freq=0.0)
    ts = toa.get_TOAs_list([t1], ephem="DE421")
    outnm = os.path.join(tmpdir, "testbaryT2.tim")
    ts.write_TOA_file(outnm, format="TEMPO")
    ts2 = toa.get_TOAs(outnm)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_roundtrip_topo_toa_Tempo2format(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    t1 = toa.TOA(t1time, obs="gbt", freq=0.0)
    ts = toa.get_TOAs_list([t1], ephem="DE421")
    outnm = os.path.join(tmpdir, "testtopoT2.tim")
    ts.write_TOA_file(outnm, format="Tempo2")
    ts2 = toa.get_TOAs(outnm)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_commenting_toas(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    t1 = toa.TOA(t1time, obs="gbt", freq=0.0)
    t2 = toa.TOA(t1time, obs="gbt", freq=0.0)
    ts = toa.get_TOAs_list([t1, t2], ephem="DE421")
    assert ts.ntoas == 2
    outnm = os.path.join(tmpdir, "testtopo.tim")
    ts.write_TOA_file(outnm)
    ts2 = toa.get_TOAs(outnm)
    assert ts2.ntoas == 2  # none should be commented
    ts.table[0]["flags"]["cut"] = "do_not_like"  # default comment flag
    ts.table[1]["flags"]["ignore"] = True  # flag with different flag
    ts.write_TOA_file(outnm)
    ts3 = toa.get_TOAs(outnm)
    assert ts3.ntoas == 1  # one should be commented
    ts.write_TOA_file(outnm, commentflag=None)
    ts4 = toa.get_TOAs(outnm)
    assert ts4.ntoas == 2  # none should be commented
    ts.write_TOA_file(outnm, commentflag="ignore")
    ts5 = toa.get_TOAs(outnm)
    assert ts5.ntoas == 1  # one should be commented


def test_roundtrip_topo_toa_TEMPOformat(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    t1 = toa.TOA(t1time, obs="gbt", freq=0.0)
    ts = toa.get_TOAs_list([t1], ephem="DE421")
    outnm = os.path.join(tmpdir, "testtopot1.tim")
    ts.write_TOA_file(outnm, format="TEMPO")
    ts2 = toa.get_TOAs(outnm)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


# Comment out because TEMPO2 distro doesn't include gmrt2gps.clk so far
# def test_roundtrip_gmrt_toa_Tempo2format(self):
#     if os.getenv("TEMPO2") is None:
#         pytest.skip("TEMPO2 evnironment variable is not set, can't run this test")
#     # Create a barycentric TOA
#     t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
#     t1 = toa.TOA(t1time, obs="gmrt", freq=0.0)
#     ts = toa.get_TOAs_list([t1], ephem="DE421")
#     ts.write_TOA_file("testgmrt.tim", format="Tempo2")
#     ts2 = toa.get_TOAs("testgmrt.tim")
#     print(ts.table, ts2.table)
#     assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
#     assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d
# def test_roundtrip_ncyobs_toa_Tempo2format(self):
#     if os.getenv("TEMPO2") is None:
#         pytest.skip("TEMPO2 evnironment variable is not set, can't run this test")
#     # Create a barycentric TOA
#     t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
#     t1 = toa.TOA(t1time, obs="ncyobs", freq=0.0)
#     ts = toa.get_TOAs_list([t1], ephem="DE421")
#     ts.write_TOA_file("testncyobs.tim", format="Tempo2")
#     ts2 = toa.get_TOAs("testncyobs.tim")
#     print(ts.table, ts2.table)
#     assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15*u.d
#     assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15*u.d
# def test_roundtrip_ncyobs_toa_TEMPOformat(self):
#     if os.getenv("TEMPO2") is None:
#         pytest.skip("TEMPO2 evnironment variable is not set, can't run this test")
#     # Create a barycentric TOA
#     t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
#     t1 = toa.TOA(t1time, obs="ncyobs", freq=0.0)
#     ts = toa.get_TOAs_list([t1], ephem="DE421")
#     # This is an observatory that can't be represented in TEMPO format
#     # so it should raise an exception
#     with pytest.raises(ValueError):
#         ts.write_TOA_file("testncyobs.tim", format="TEMPO")
