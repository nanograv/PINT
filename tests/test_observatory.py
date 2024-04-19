import io
import os
import json

import astropy.units as u
import contextlib
import numpy as np
import pytest
from astropy import units as u

from pint.pulsar_mjd import Time
from pinttestdata import datadir as testdatadir

import pint.observatory
from pint.observatory import (
    NoClockCorrections,
    Observatory,
    get_observatory,
    compare_t2_observatories_dat,
    compare_tempo_obsys_dat,
)
from pint.pulsar_mjd import Time
import pint.observatory.topo_obs
from pint.observatory.topo_obs import (
    TopoObs,
    load_observatories,
)
from collections import defaultdict

tobs = ["aro", "ao", "chime", "drao"]


@pytest.fixture
def test_time():
    return Time(np.linspace(55000, 58000, num=100), scale="utc", format="pulsar_mjd")


@pytest.mark.parametrize("tobs", tobs)
def test_get_obs(tobs):
    site = get_observatory(
        tobs, include_gps=False, include_bipm=True, bipm_version="BIPM2015"
    )
    assert site


@pytest.mark.parametrize("tobs", tobs)
def test_different_bipm(tobs):
    site = get_observatory(
        tobs, include_gps=False, include_bipm=True, bipm_version="BIPM2019"
    )
    assert site


@pytest.mark.parametrize("tobs", tobs)
def test_clock_corr_shape(tobs, test_time):
    site = get_observatory(
        tobs, include_gps=True, include_bipm=True, bipm_version="BIPM2015"
    )
    clock_corr = site.clock_corrections(test_time)
    assert len(clock_corr) == len(test_time)
    clock_corr1 = site.clock_corrections(test_time[0])
    assert clock_corr1.shape == ()


@pytest.mark.parametrize("tobs", tobs)
def test_get_TDBs(tobs, test_time):
    site = get_observatory(
        tobs, include_gps=True, include_bipm=True, bipm_version="BIPM2015"
    )
    # Test default TDB calculation
    tdbs = site.get_TDBs(test_time)
    assert len(tdbs) == len(test_time)
    tdb1 = site.get_TDBs(test_time[0])
    assert tdb1.shape == (1,)


@pytest.mark.parametrize("tobs", tobs)
def test_get_TDBs_ephemeris(tobs, test_time):
    site = get_observatory(
        tobs, include_gps=True, include_bipm=True, bipm_version="BIPM2015"
    )

    # Test TDB calculation from ephemeris
    tdbs = site.get_TDBs(test_time, method="ephemeris", ephem="de430t")
    assert len(tdbs) == len(test_time)
    tdb1 = site.get_TDBs(test_time[0], method="ephemeris", ephem="de430t")
    assert tdb1.shape == (1,)


@pytest.mark.parametrize("tobs", tobs)
def test_positions_shape(tobs, test_time):
    site = get_observatory(
        tobs, include_gps=True, include_bipm=True, bipm_version="BIPM2015"
    )
    posvel = site.posvel(test_time, ephem="de436")
    assert posvel.pos.shape == (3, len(test_time))
    assert posvel.vel.shape == (3, len(test_time))


def test_wrong_TDB_method_raises(test_time):
    site = get_observatory(
        "ao", include_gps=True, include_bipm=True, bipm_version="BIPM2015"
    )
    with pytest.raises(ValueError):
        site.get_TDBs(test_time, method="ephemeris")
    with pytest.raises(ValueError):
        site.get_TDBs(test_time, method="Unknown_method")
    with pytest.raises(ValueError):
        site.get_TDBs(test_time, method="ephemeris", ephem="de436")


def test_wrong_name():
    with pytest.raises(KeyError):
        get_observatory("Wrong_name")


@pytest.fixture
def sandbox():
    class Sandbox:
        pass

    o = Sandbox()
    e = os.environ.copy()

    with contextlib.suppress(KeyError):
        del os.environ["PINT_OBS_OVERRIDE"]
    reg = pint.observatory.Observatory._registry.copy()
    try:
        yield o
    finally:
        os.environ = e
        pint.observatory.Observatory._registry = reg


@pytest.mark.parametrize(
    "observatory", list(pint.observatory.Observatory._registry.keys())
)
def test_can_try_to_compute_corrections(observatory):
    # Many of these should emit warnings
    get_observatory(observatory).clock_corrections(Time(57600, format="mjd"))


good_observatories = [
    "gbt",
    "ao",
    "vla",
    "jodrell",
    "jbroach",
    "jbdfb",
    "wsrt",
    "parkes",
]


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
        )

        with pytest.raises(NoClockCorrections):
            o.clock_corrections(Time(57600, format="mjd"), limits="error")
    finally:
        Observatory._registry = r


def test_no_clock_means_no_corrections():
    r = Observatory._registry.copy()
    try:
        o = TopoObs(
            "arecibo_bogus",
            itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
            include_gps=False,
            include_bipm=False,
        )

        assert (
            o.clock_corrections(Time(57600, format="mjd"), limits="error").to_value(u.s)
            == 0
        )
    finally:
        Observatory._registry = r


@pytest.mark.parametrize("var", ["TEMPO", "TEMPO2"])
def test_missing_env_raises(var):
    r = Observatory._registry.copy()
    e = os.environ.copy()
    try:
        os.environ.pop(var, None)
        o = TopoObs(
            "arecibo_bogus",
            clock_file="nonexistent.dat",
            clock_dir=var,
            itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
        )
        with pytest.raises(NoClockCorrections):
            o.clock_corrections(Time(57600, format="mjd"))
    finally:
        Observatory._registry = r
        os.environ = e


def test_observatories_registered():
    assert len(pint.observatory.Observatory._registry) > 5


def test_gbt_registered():
    get_observatory("gbt")


def test_is_gbt_still_ok():
    gbt = get_observatory("gbt")
    assert gbt.location.y < 0


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
        assert newgbt.location.y > 0
        assert newgbt.location.y != gbt_orig.location.y


def test_list_last_correction_mjds_runs():
    pint.observatory.list_last_correction_mjds()


def test_TopoObs_EarthLocation(sandbox):
    gbt_orig = get_observatory("gbt")
    gbt_new = TopoObs(name="gbt_new", location=gbt_orig.location)
    assert gbt_orig.separation(gbt_new) < 1 * u.cm


def test_json_observatory_output(sandbox):
    gbt_orig = get_observatory("gbt")
    load_observatories(io.StringIO(gbt_orig.get_json()), overwrite=True)
    gbt_reload = get_observatory("gbt")

    for p in gbt_orig.__dict__:
        if p not in ["_clock", "_aliases"]:
            assert getattr(gbt_orig, p) == getattr(gbt_reload, p)

        assert set(gbt_orig._aliases) == set(gbt_reload._aliases)


def test_json_observatory_input_latlon(sandbox):
    gbt_orig = get_observatory("gbt")
    gbt_dict = gbt_orig.get_dict()
    # remove ITRF
    del gbt_dict["gbt"]["itrf_xyz"]
    # add in geodetic
    gbt_dict["gbt"]["lat"] = gbt_orig.location.lat.value
    gbt_dict["gbt"]["lon"] = gbt_orig.location.lon.value
    gbt_dict["gbt"]["height"] = gbt_orig.location.height.value
    load_observatories(io.StringIO(json.dumps(gbt_dict)), overwrite=True)
    gbt_reload = get_observatory("gbt")

    for p in gbt_orig.__dict__:
        if p not in ["location", "_clock"]:
            # everything else should be identical
            # but fix the order if it is a list
            if isinstance(getattr(gbt_orig, p), list):
                assert set(getattr(gbt_orig, p)) == set(getattr(gbt_reload, p))
            else:
                assert getattr(gbt_orig, p) == getattr(gbt_reload, p)
    # check distance separately to allow for precision
    distance = gbt_orig.separation(gbt_reload)
    assert distance < 1 * u.m


def test_json_observatory_input_latlon_and_itrf_giveserror(sandbox):
    gbt_orig = get_observatory("gbt")
    gbt_dict = gbt_orig.get_dict()
    # add in geodetic
    gbt_dict["gbt"]["lat"] = gbt_orig.location.lat.value
    gbt_dict["gbt"]["lon"] = gbt_orig.location.lon.value
    gbt_dict["gbt"]["height"] = gbt_orig.location.height.value
    with pytest.raises(ValueError):
        load_observatories(io.StringIO(json.dumps(gbt_dict)), overwrite=True)

    del gbt_dict["gbt"]["itrf_xyz"]
    del gbt_dict["gbt"]["lat"]
    with pytest.raises(ValueError):
        load_observatories(io.StringIO(json.dumps(gbt_dict)), overwrite=True)


def test_valid_past_end():
    o = pint.observatory.get_observatory("jbroach")
    o.last_clock_correction_mjd()
    o.clock_corrections(o._clock[0].time[-1] + 1 * u.d, limits="error")


def test_names_and_aliases():
    na = Observatory.names_and_aliases()
    assert isinstance(na, dict) and isinstance(na["gbt"], list)


def test_compare_t2_observatories_dat():
    s = compare_t2_observatories_dat(testdatadir)
    assert isinstance(s, defaultdict)


def test_compare_tempo_obsys_dat():
    s = compare_tempo_obsys_dat(testdatadir / "observatory")
    assert isinstance(s, defaultdict)


def test_ssb_obs():
    ssb = Observatory.get("@")
    assert not ssb.include_bipm and not ssb.include_gps

    ssb = get_observatory("@")
    assert not ssb.include_bipm and not ssb.include_gps

    # get_observatory changes the state of the registered
    # Observatory objects. So this needs to be repeated.
    ssb = Observatory.get("@")
    assert not ssb.include_bipm and not ssb.include_gps
