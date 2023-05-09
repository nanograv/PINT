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
from pint import toa, observatory
from datetime import datetime
from astropy.time import Time
from astropy import units as u
import pint


def test_roundtrip_bary_toa_Tempo2format(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="tdb")
    ts = toa.get_TOAs_array(t1time, "Barycenter", freqs=0, ephem="DE421")
    outnm = os.path.join(tmpdir, "testbaryT.tim")
    ts.write_TOA_file(outnm, format="Tempo2")
    ts2 = toa.get_TOAs(outnm)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_roundtrip_bary_toa_TEMPOformat(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="tdb")
    ts = toa.get_TOAs_array(t1time, "Barycenter", freqs=0, ephem="DE421")
    outnm = os.path.join(tmpdir, "testbaryT2.tim")
    ts.write_TOA_file(outnm, format="TEMPO")
    ts2 = toa.get_TOAs(outnm)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_roundtrip_topo_toa_Tempo2format(tmpdir):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    ts = toa.get_TOAs_array(t1time, "gbt", freqs=0, ephem="DE421")
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
    ts = toa.get_TOAs_array(t1time, "gbt", freqs=0, ephem="DE421")
    outnm = os.path.join(tmpdir, "testtopot1.tim")
    ts.write_TOA_file(outnm, format="TEMPO")
    ts2 = toa.get_TOAs(outnm)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_roundtrip_gmrt_toa_Tempo2format(tmp_path):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    ts = toa.get_TOAs_array(t1time, "gbt", freqs=0, ephem="DE421")
    ts.write_TOA_file(tmp_path / "testgmrt.tim", format="Tempo2")
    ts2 = toa.get_TOAs(tmp_path / "testgmrt.tim")
    print(ts.table, ts2.table)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_roundtrip_ncyobs_toa_Tempo2format(tmp_path):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    ts = toa.get_TOAs_array(t1time, "ncyobs", freqs=0, ephem="DE421")
    ts.write_TOA_file(tmp_path / "testncyobs.tim", format="Tempo2")
    ts2 = toa.get_TOAs(tmp_path / "testncyobs.tim")
    print(ts.table, ts2.table)
    assert np.abs(ts.table["mjd"][0] - ts2.table["mjd"][0]) < 1.0e-15 * u.d
    assert np.abs(ts.table["tdb"][0] - ts2.table["tdb"][0]) < 1.0e-15 * u.d


def test_roundtrip_ncyobs_toa_TEMPOformat(tmp_path):
    # Create a barycentric TOA
    t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    ts = toa.get_TOAs_array(t1time, "ncyobs", freqs=0, ephem="DE421")
    # This is an observatory that can't be represented in TEMPO format
    # so it should raise an exception
    with pytest.raises(ValueError):
        ts.write_TOA_file(tmp_path / "testncyobs.tim", format="TEMPO")


def test_write_pn_nan():
    # NaN should not get written as flag in tim file
    model = get_model(datadir / "NGC6440E.par")
    # read and add pulse numbers
    t = toa.get_TOAs(datadir / "NGC6440E.tim", model=model)
    t.compute_pulse_numbers(model)
    t["pulse_number"][10] = np.nan
    f = StringIO()
    t.write_TOA_file(f)
    f.seek(0)
    # contents = f.read()
    for line in f.readlines():
        if toa._toa_format(line) != "Comment":
            assert "nan" not in line.split()
    f.seek(0)
    t2 = toa.get_TOAs(f)
    assert np.isnan(t2["pulse_number"][10])


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


def test_tim_writing_order(tmp_path):
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


@pytest.mark.parametrize("format", ["Tempo2", "Princeton"])
def test_format_toa_line(format):
    toatime = Time(datetime.now())
    toaerr = u.Quantity(1e-6, "s")
    freq = u.Quantity(1400, "MHz")
    obs = observatory.get_observatory("ao")
    dm = 1 * pint.dmu

    toa_line = toa.format_toa_line(
        toatime,
        toaerr,
        freq,
        obs,
        dm=dm,
        name="unk",
        flags={"-foo": 1, "bar": "ee", "spam": u.Quantity(3, "s")},
        format=format,
        alias_translation=None,
    )

    assert isinstance(toa_line, str) and len(toa_line) > 0


def test_format_toa_line_bad_fmt():
    toatime = Time(datetime.now())
    toaerr = u.Quantity(1e-6, "s")
    freq = u.Quantity(1400, "MHz")
    obs = observatory.get_observatory("ao")
    dm = 1 * pint.dmu

    with pytest.raises(ValueError):
        toa_line = toa.format_toa_line(
            toatime,
            toaerr,
            freq,
            obs,
            dm=dm,
            name="unk",
            flags={"-foo": 1, "bar": 2, "spam": u.Quantity(3, "s")},
            format="bla",
            alias_translation=None,
        )
