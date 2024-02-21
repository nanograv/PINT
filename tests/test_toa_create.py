import numpy as np
from astropy import units as u
from pint import pulsar_mjd
from pint import toa
from pint.models import get_model
from astropy import time
from pint.fermi_toas import load_Fermi_TOAs, get_Fermi_TOAs
from pint.observatory.satellite_obs import get_satellite_observatory
from pinttestdata import datadir
import copy
import io
import pytest


@pytest.mark.parametrize(
    "t",
    [
        pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd"),
        np.array([55000, 56000]),
        (np.array([55000, 56000]), np.array([0, 0])),
    ],
)
@pytest.mark.parametrize("errors", [1, 1.0 * u.us, np.array([1.0, 2.3]) * u.us])
@pytest.mark.parametrize("freqs", [1400, 400 * u.MHz, np.array([1400, 1500]) * u.MHz])
def test_toas_fromarray(t, errors, freqs):
    obs = "gbt"
    kwargs = {"name": "data"}
    flags = [{"type": "good"}, {"type": "bad"}]

    toas = toa.get_TOAs_array(t, obs, errors=errors, freqs=freqs, flags=flags, **kwargs)


@pytest.mark.parametrize(
    "t", [55000, (55000, 0.0), pulsar_mjd.Time(55000, scale="utc", format="pulsar_mjd")]
)
def test_toas_fromscalar(t):
    toas = toa.get_TOAs_array(t, "gbt")


@pytest.mark.parametrize(
    "t",
    [
        pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd"),
        np.array([55000, 56000]),
        (np.array([55000, 56000]), np.array([0, 0])),
    ],
)
@pytest.mark.parametrize("errors", [1, 1.0 * u.us, np.array([1.0, 2.3]) * u.us])
@pytest.mark.parametrize("freqs", [1400, 400 * u.MHz, np.array([1400, 1500]) * u.MHz])
def test_toas_compare(t, errors, freqs):
    obs = "gbt"
    kwargs = {"name": "data"}
    flags = [{"type": "good"}, {"type": "bad"}]
    toas = toa.get_TOAs_array(t, obs, errors=errors, freqs=freqs, flags=flags, **kwargs)

    combined_flags = copy.deepcopy(flags)
    for flag in combined_flags:
        for k, v in kwargs.items():
            flag[k] = v
    errors = np.atleast_1d(errors)
    freqs = np.atleast_1d(freqs)
    if len(errors) < len(t):
        errors = np.repeat(errors, len(t))
    if len(freqs) < len(t):
        freqs = np.repeat(freqs, len(t))
    toalist = (
        [
            toa.TOA((tt0, tt1), obs=obs, error=e, freq=fr, flags=f)
            for tt0, tt1, e, f, fr in zip(t[0], t[1], errors, combined_flags, freqs)
        ]
        if isinstance(t, tuple)
        else [
            toa.TOA(tt, obs=obs, error=e, freq=fr, flags=f)
            for tt, e, f, fr in zip(t, errors, combined_flags, freqs)
        ]
    )
    toas_fromlist = toa.get_TOAs_list(toalist)
    assert np.all(toas.table == toas_fromlist.table)


@pytest.mark.parametrize(
    "t",
    [
        pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd"),
        np.array([55000, 56000]),
        (np.array([55000, 56000]), np.array([0, 0])),
    ],
)
@pytest.mark.parametrize("errors", [1, 1.0 * u.us, np.array([1.0, 2.3]) * u.us])
@pytest.mark.parametrize("freqs", [1400, 400 * u.MHz, np.array([1400, 1500]) * u.MHz])
def test_toas_tolist(t, errors, freqs):
    obs = "gbt"
    kwargs = {"name": "data"}
    flags = [{"type": "good"}, {"type": "bad"}]
    toas = toa.get_TOAs_array(t, obs, errors=errors, freqs=freqs, flags=flags, **kwargs)

    combined_flags = copy.deepcopy(flags)
    for flag in combined_flags:
        for k, v in kwargs.items():
            flag[k] = v
    errors = np.atleast_1d(errors)
    freqs = np.atleast_1d(freqs)
    if len(errors) < len(t):
        errors = np.repeat(errors, len(t))
    if len(freqs) < len(t):
        freqs = np.repeat(freqs, len(t))
    if not isinstance(t, tuple):
        toalist = [
            toa.TOA(tt, obs=obs, error=e, freq=fr, flags=f)
            for tt, e, f, fr in zip(t, errors, combined_flags, freqs)
        ]
    else:
        toalist = [
            toa.TOA((tt0, tt1), obs=obs, error=e, freq=fr, flags=f)
            for tt0, tt1, e, f, fr in zip(t[0], t[1], errors, combined_flags, freqs)
        ]
    toas_tolist = toas.to_TOA_list()
    # for i in range(len(toalist)):
    #     assert toalist[i] == toas_tolist[i], f"{toalist[i]} != {toas_tolist[i]}"
    # depending on precision they should be equal, but if they aren't then just check the MJDs
    assert np.all([t1 == t2 for (t1, t2) in zip(toalist, toas_tolist)]) or np.allclose(
        [x.mjd.mjd for x in toalist], [x.mjd.mjd for x in toas_tolist]
    )


def test_kwargs_as_flags():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    kwargs = {"name": "data", "energies": [1, 2]}
    flags = [{"type": "good"}, {"type": "bad"}]
    toas = toa.get_TOAs_array(t, obs, flags=flags, **kwargs)
    assert np.all(toas["energies"] == np.array(["1", "2"]))


@pytest.mark.parametrize("errors", [2 * u.m, np.array([1, 2, 3]) * u.us])
def test_toas_failure_error(errors):
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    with pytest.raises((AttributeError, u.UnitConversionError)):
        toas = toa.get_TOAs_array(t, obs, errors=errors)


@pytest.mark.parametrize("freqs", [2 * u.m, np.array([1000, 2000, 3000]) * u.MHz])
def test_toas_failure_freqs(freqs):
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    with pytest.raises((AttributeError, u.UnitConversionError)):
        toas = toa.get_TOAs_array(t, obs, freqs=freqs)


def test_toas_failure_flags():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    with pytest.raises(AttributeError):
        toas = toa.get_TOAs_array(t, obs, flags=[{"a": "b"}, {"c": "d"}, {"e": "f"}])


def test_toas_failure_scale():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    with pytest.raises(ValueError):
        toas = toa.get_TOAs_array(t, obs, scale="tt")


def test_toas_model():
    par = """PSR    J0613-0200
      LAMBDA 93.7990065496191  1 0.0000000158550
      BETA   -25.4071326875232  1 0.0000000369013
      PMLAMBDA 2.1192  1 0.0174
      PMBETA -10.3422  1              0.0433
      PX 0.9074  1              0.1509
      F0 326.6005670972169810  1  0.0000000000066373
      F1 -1.022985317101D-15  1  6.219122230955D-20
      PEPOCH        54890.000000
      DM         38.778683
      EPHEM DE440
      CLOCK TT(BIPM2019)
      """
    m = get_model(io.StringIO(par))
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    toas = toa.get_TOAs_array(t, obs, model=m, include_bipm=None, bipm_version=None)
    assert toas.ephem == "DE440"
    assert toas.clock_corr_info["bipm_version"] == "BIPM2019"
    assert toas.clock_corr_info["include_bipm"]

    par = """PSR    J0613-0200
      LAMBDA 93.7990065496191  1 0.0000000158550
      BETA   -25.4071326875232  1 0.0000000369013
      PMLAMBDA 2.1192  1 0.0174
      PMBETA -10.3422  1              0.0433
      PX 0.9074  1              0.1509
      F0 326.6005670972169810  1  0.0000000000066373
      F1 -1.022985317101D-15  1  6.219122230955D-20
      PEPOCH        54890.000000
      DM         38.778683
      EPHEM DE440
      CLOCK TT(TAI)
      """
    m = get_model(io.StringIO(par))
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    toas = toa.get_TOAs_array(t, obs, model=m, include_bipm=None, bipm_version=None)
    assert toas.ephem == "DE440"
    assert not toas.clock_corr_info["include_bipm"]


def test_toas_clockflag_adjust():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    toas = toa.get_TOAs_array(t, obs, to="1.2")
    toas2 = toa.get_TOAs_array(t, obs)

    assert np.allclose(toas["mjd_float"] - toas2["mjd_float"], 1.2 / 86400)


def test_toas_clockflag_allcolumns():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    toas = toa.get_TOAs_array(t, obs, to="1.2")

    assert np.allclose(np.array([x.mjd for x in toas["mjd"]]) - toas["mjd_float"], 0)


def test_toas_fermi():
    eventfileraw = datadir / "J0030+0451_w323_ft1weights.fits"
    ft2file = datadir / "lat_spacecraft_weekly_w323_p202_v001.fits"
    get_satellite_observatory("Fermi", ft2file, overwrite=True)

    tl = load_Fermi_TOAs(eventfileraw, weightcolumn="PSRJ0030+0451")
    ts = toa.get_TOAs_list(
        tl, include_gps=False, include_bipm=False, planets=False, ephem="DE405"
    )

    t = time.Time([toa.mjd for toa in tl], scale="tt")
    flags = [toa.flags for toa in tl]
    ts2 = toa.get_TOAs_array(
        t,
        "fermi",
        include_gps=False,
        include_bipm=False,
        planets=False,
        ephem="DE405",
        flags=flags,
    )
    assert np.all(ts["mjd"] == ts2["mjd"])


def test_toas_fermi_notoalist():
    eventfileraw = datadir / "J0030+0451_w323_ft1weights.fits"
    ft2file = datadir / "lat_spacecraft_weekly_w323_p202_v001.fits"
    get_satellite_observatory("Fermi", ft2file, overwrite=True)

    tl = load_Fermi_TOAs(eventfileraw, weightcolumn="PSRJ0030+0451")
    toas = toa.get_TOAs_list(
        tl, include_gps=False, include_bipm=False, planets=False, ephem="DE405"
    )

    toas2 = get_Fermi_TOAs(
        eventfileraw,
        weightcolumn="PSRJ0030+0451",
        planets=False,
        ephem="DE405",
    )
    assert np.all(toas["mjd"] == toas2["mjd"])
    assert np.all(toas.table == toas2.table)


def test_geocenter():
    toas = toa.get_TOAs(
        io.StringIO(
            """pint 0.000000 57932.5446286608075925 4.321 stl_geo  -format Tempo2 -creator nicer -mjdtt 57932.5454294 -tely -2783.63069101 -telx 6119.42973459 -telz -890.7794845 -nsrc 18.34 -obsid 70020101 -t NICER -vy 4.04867262 -vx 2.69970132 -dphi 0.20496 -nbkg 430.66 -vz 5.92980989 -exposure 1543.0
pint 0.000000 57933.6326461481560532 2.422 stl_geo  -format Tempo2 -creator nicer -mjdtt 57933.6334469 -tely -4545.24145649 -telx 3799.22279181 -telz -3304.56529985 -nsrc 101.96 -obsid 70020102 -t NICER -vy 1.48403228 -vx 5.86477721 -dphi 0.20355 -nbkg 2394.04 -vz 4.70930058 -exposure 8100.71
"""
        )
    )
    toas2 = toa.get_TOAs(
        io.StringIO(
            """pint 0.000000 57932.5446286608075925 4.321 spacecraft  -format Tempo2 -creator nicer -mjdtt 57932.5454294 -tely -2783.63069101 -telx 6119.42973459 -telz -890.7794845 -nsrc 18.34 -obsid 70020101 -t NICER -vy 4.04867262 -vx 2.69970132 -dphi 0.20496 -nbkg 430.66 -vz 5.92980989 -exposure 1543.0
pint 0.000000 57933.6326461481560532 2.422 spacecraft  -format Tempo2 -creator nicer -mjdtt 57933.6334469 -tely -4545.24145649 -telx 3799.22279181 -telz -3304.56529985 -nsrc 101.96 -obsid 70020102 -t NICER -vy 1.48403228 -vx 5.86477721 -dphi 0.20355 -nbkg 2394.04 -vz 4.70930058 -exposure 8100.71
"""
        )
    )
    assert np.all(toas == toas2)


def test_toa_wb():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    pp_dm = np.array([1, 2]) * u.pc / u.cm**3
    pp_dme = np.array([0.1, 0.1]) * u.pc / u.cm**3
    toas = toa.get_TOAs_array(t, obs, pp_dm=pp_dm.value, pp_dme=pp_dme.value)
    assert toas.wideband


def test_toa_add_is_merge():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    t2 = pulsar_mjd.Time(np.array([56500, 57000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    toas = toa.get_TOAs_array(t, obs)
    toas2 = toa.get_TOAs_array(t2, obs)
    toasout = toa.merge_TOAs([toas, toas2])
    toasout_add = toas + toas2
    assert np.all(toasout.table == toasout_add.table)


def test_toa_merge():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    t2 = pulsar_mjd.Time(np.array([56500, 57000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    toas = toa.get_TOAs_array(t, obs)
    toas2 = toa.get_TOAs_array(t2, obs)
    toasout = toa.merge_TOAs([toas, toas2])
    toas.merge(toas2)
    assert np.all(toasout.table == toas.table)


def test_toa_iadd_is_merge():
    t = pulsar_mjd.Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    t2 = pulsar_mjd.Time(np.array([56500, 57000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"
    toas = toa.get_TOAs_array(t, obs)
    startid = id(toas)
    toas2 = toa.get_TOAs_array(t2, obs)
    toasout = toa.merge_TOAs([toas, toas2])
    toas += toas2
    assert np.all(toasout.table == toas.table)
    assert startid == id(toas)
