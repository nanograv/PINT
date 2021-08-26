from io import StringIO
import os
import os.path
import pytest

from pint.models import get_model
from pint import toa, simulation
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
    ts.table[0]["flags"]["cut"] = "do_not_like"  # cut flag
    ts.table[1]["flags"]["ignore"] = str(1)  # ignore flag
    ts.write_TOA_file(outnm)
    ts3 = toa.get_TOAs(outnm)  # defaut is to not comment
    assert ts3.ntoas == 2  # none should be commented by default
    ts.write_TOA_file(outnm, commentflag="cut")
    ts4 = toa.get_TOAs(outnm)
    assert ts4.ntoas == 1  # one should be commented
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

simplepar = """
PSR              1748-2021E
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
EPHEM               DE436
CLK              TT(BIPM2017)
UNITS               TDB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      Y
DILATEFREQ          N
"""


def test_tim_writing_order():
    m = get_model(StringIO(simplepar))
    fakes = [
        simulation.make_fake_toas_uniform(55000, 56000, 5, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(55000, 56000, 5, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(55000, 56000, 5, model=m, obs="@"),
    ]
    toas = toa.merge_TOAs(fakes)
    toas.table["index"][np.argsort(toas.table["tdbld"])] = np.arange(len(toas))

    o = StringIO()
    toas.write_TOA_file(o, order_by_index=False)
    toas.write_TOA_file("test.tim", order_by_index=False)
    obs = [
        ln.split()[4]
        for ln in o.getvalue().split("\n")[1:]
        if (ln and not ln.startswith("C "))
    ]
    assert obs[0] == obs[1] == obs[2] == obs[3]

    o = StringIO()
    toas.write_TOA_file(o, order_by_index=True)
    obs = [
        ln.split()[4]
        for ln in o.getvalue().split("\n")[1:]
        if (ln and not ln.startswith("C "))
    ]
    assert obs[0] == obs[3] == obs[6] == obs[9]
    assert obs[0] != obs[1]
