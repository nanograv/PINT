import numpy as np
from astropy import units as u, constants as c
from pint import pulsar_mjd
from pint import toa
from pint.models import get_model
from astropy import time
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
    toas_fromlist = toa.get_TOAs_list(toalist)
    assert np.all(toas.table == toas_fromlist.table)


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
