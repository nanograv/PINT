import decimal
import re
import sys
import warnings
from datetime import datetime
from decimal import Decimal

import erfa
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from hypothesis import assume, example, given, settings, HealthCheck, target
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
    one_of,
    sampled_from,
)
from numpy.testing import assert_array_equal

from pint.pulsar_mjd import (
    data2longdouble,
    day_frac,
    jds_to_mjds,
    jds_to_mjds_pulsar,
    longdouble2str,
    mjds_to_jds,
    mjds_to_jds_pulsar,
    mjds_to_str,
    safe_kind_conversion,
    str2longdouble,
    str_to_mjds,
    time_from_longdouble,
    time_from_mjd_string,
    time_to_longdouble,
    time_to_mjd_string,
    two_sum,
)
from pint.utils import PosVel

time_eps = (np.finfo(float).eps * u.day).to(u.ns)


leap_sec_mjds = {
    41499,
    41683,
    42048,
    42413,
    42778,
    43144,
    43509,
    43874,
    44239,
    44786,
    45151,
    45516,
    46247,
    47161,
    47892,
    48257,
    48804,
    49169,
    49534,
    50083,
    50630,
    51179,
    53736,
    54832,
    56109,
    57204,
    57754,
}
leap_sec_days = {d - 1 for d in leap_sec_mjds}
near_leap_sec_days = list(
    sorted([d - 1 for d in leap_sec_days] + [d + 1 for d in leap_sec_days])
)


@composite
def possible_leap_sec_days(draw):
    y = draw(integers(2017, 2050))
    if s := draw(booleans()):
        m = Time(datetime(y, 6, 30, 0, 0, 0), scale="tai").mjd
    else:
        m = Time(datetime(y, 12, 31, 0, 0, 0), scale="tai").mjd
    return int(np.round(m))


@composite
def leap_sec_day_mjd(draw):
    i = draw(sampled_from(sorted(leap_sec_days)))
    f = draw(floats(0, 1, allow_nan=False))
    return (i, f)


@composite
def normal_mjd(draw):
    i = draw(
        one_of(
            integers(40000, 70000),
            sampled_from(near_leap_sec_days),
            possible_leap_sec_days(),
        )
    )
    assume(i not in leap_sec_days)
    f = draw(floats(0, 1, allow_nan=False))
    return (i, f)


@composite
def reasonable_mjd(draw):
    return draw(one_of(normal_mjd(), leap_sec_day_mjd()))


@composite
def unreasonable_mjd(draw):
    i = draw(integers(-3000000, 3000000))
    f = draw(floats(-2, 2, allow_nan=False))
    return (i, f)


@composite
def any_mjd(draw):
    return draw(one_of(normal_mjd(), leap_sec_day_mjd(), unreasonable_mjd()))


# longdouble2str, str2longdouble


@given(any_mjd())
def test_str_roundtrip_is_exact(i_f):
    i, f = i_f
    ld = np.longdouble(i) + np.longdouble(f)
    assert ld == np.longdouble(str(ld))


@given(any_mjd())
def test_longdouble_str_roundtrip_is_exact(i_f):
    i, f = i_f
    ld = np.longdouble(i) + np.longdouble(f)
    assert ld == str2longdouble(longdouble2str(ld))


@given(any_mjd())
def test_longdouble2str_same_as_str_and_repr(i_f):
    i, f = i_f
    ld = np.longdouble(i) + np.longdouble(f)
    assert longdouble2str(ld) == str(ld)
    assert longdouble2str(ld) == repr(ld)


@pytest.mark.parametrize("s", ["1.0.2", "1.0.", "1.0e1e0", "twelve"])
def test_str2longdouble_raises_valueerror(s):
    with pytest.raises(ValueError):
        str2longdouble(s)


@pytest.mark.parametrize("s, v", [("1.0d2", 1.0e2), ("NAN", np.nan), ("INF", np.inf)])
def test_str2longdouble_handles_unusual_input(s, v):
    if np.isnan(v):
        assert np.isnan(str2longdouble(s))
    else:
        assert str2longdouble(s) == v


@pytest.mark.parametrize("s", [1.0, 1, None, {}, [], re.compile("fo+")])
def test_str2longdouble_rejects_non_strings(s):
    with pytest.raises(TypeError):
        str2longdouble(s)


@pytest.mark.skipif(
    sys.version_info < (3,), reason="python 2 bytes objects are just strings"
)
@pytest.mark.parametrize("s", [b"", b"1", b"1.34", b"abd"])
def test_str2longdouble_rejects_bytes(s):
    with pytest.raises(TypeError):
        str2longdouble(s)


@pytest.mark.parametrize(
    "d, ld",
    [
        ("1", 1),
        (1.0, 1.0),
        (np.float32(1.0), 1.0),
        (np.longdouble(1.0), 1.0),
        (np.array([1]), np.array([1.0])),
        (np.array([[1]]), np.array([[1.0]])),
    ],
)
def test_data2longdouble_accepts_types(d, ld):
    assert data2longdouble(d) == ld


# data2longdouble


@pytest.mark.parametrize(
    "a",
    [
        [1],
        [1, 2, 3],
        [1.5, 2],
        np.ones(5, dtype=np.longdouble) + np.finfo(np.longdouble).eps,
        np.random.randn(2, 3, 4),
    ],
)
def test_data2longdouble_converts_arrays(a):
    assert_array_equal(data2longdouble(a), np.asarray(a, dtype=np.longdouble))


# astropy.time.Time construction (?!)


@given(reasonable_mjd())
@example(i_f=(65536, 3.552713678800502e-15))
@example(i_f=(43143, 9.313492199680697e-10))
@example(i_f=(40000, -4.440892098500627e-16))
@example(i_f=(40000, -4.440892098500627e-16))
@example(i_f=(65536, 3.637978807091714e-12))  # scale="tdb"
@pytest.mark.parametrize("scale", ["tai", "tt", "tdb"])
def test_time_construction_jds_exact(scale, i_f):
    i, f = i_f
    jd1, jd2 = day_frac(i + erfa.DJM0, f)
    t = Time(val=jd1, val2=jd2, format="jd", scale=scale)
    jd1_t, jd2_t = day_frac(t.jd1, t.jd2)
    assert (jd1, jd2) == (jd1_t, jd2_t)


@given(reasonable_mjd())
@example(i_f=(40000, -4.440892098500627e-16))
@example(i_f=(43143, 9.313492199680697e-10))
@example(i_f=(65536, 3.637978807091714e-12))
# @example(i_f=(65536, 3.552713678800502e-15))
def test_time_construction_mjds_preserved(i_f):
    i, f = i_f
    t = Time(val=i, val2=f, format="mjd", scale="tai")
    jd1, jd2 = day_frac(i + erfa.DJM0, f)
    jd1_t, jd2_t = day_frac(t.jd1, t.jd2)
    assert (abs((jd1 - jd1_t) + (jd2 - jd2_t)) * u.day).to(u.ns) < 1 * u.ns


@given(reasonable_mjd())
@example(i_f=(40000, -4.440892098500627e-16))
@example(i_f=(43143, 9.313492199680697e-10))
@example(i_f=(65536, 3.637978807091714e-12))
@pytest.mark.parametrize("scale", ["tai", "tt", "tdb"])
def test_time_construction_mjd_versus_jd(scale, i_f):
    i, f = i_f
    jd1, jd2 = day_frac(i + erfa.DJM0, f)
    t = Time(val=jd1, val2=jd2, format="jd", scale=scale)
    t2 = Time(val=i, val2=f, format="mjd", scale=scale)
    assert abs(t - t2).to(u.ns) < 1 * u.ns


# time_to_longdouble, time_from_longdouble


@given(reasonable_mjd())
@example(i_f=(43143, 9.313492199680697e-10))
@example(i_f=(40000, -4.440892098500627e-16))
@example(i_f=(65536, 3.637978807091714e-12))
@pytest.mark.parametrize("scale", ["tai", "tt", "tdb"])
def test_time_to_longdouble_via_jd(scale, i_f):
    i, f = i_f
    jd1, jd2 = day_frac(i + erfa.DJM0, f)
    t = Time(val=jd1, val2=jd2, format="jd", scale=scale)
    ld = np.longdouble(i) + np.longdouble(f)
    assert (abs(time_to_longdouble(t) - ld) * u.day).to(u.ns) < 1 * u.ns


@given(reasonable_mjd())
@example(i_f=(43143, 9.313492199680697e-10))
@example(i_f=(40000, -4.440892098500627e-16))
@example(i_f=(65536, 3.637978807091714e-12))
@pytest.mark.parametrize("scale", ["tai", "tt", "tdb"])
def test_time_to_longdouble(scale, i_f):
    i, f = i_f
    t = Time(val=i, val2=f, format="mjd", scale=scale)
    ld = np.longdouble(i) + np.longdouble(f)
    assert (abs(time_to_longdouble(t) - ld) * u.day).to(u.ns) < 1 * u.ns


@pytest.mark.xfail
@given(reasonable_mjd())
@example(i_f=(43143, 9.313492199680697e-10))  # format="pulsar_mjd"
@example(i_f=(40000, -4.440892098500627e-16))  # format="pulsar_mjd"
@example(i_f=(65536, 3.637978807091714e-12))
@example(i_f=(40000, -4.440892098500627e-16))  # format="mjd"
@example(i_f=(42710, 0.45015659432648014))
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_to_longdouble_utc(format, scale, i_f):
    i, f = i_f
    t = Time(val=i, val2=f, format=format, scale="utc")
    ld = np.longdouble(i) + np.longdouble(f)
    assert_quantity_allclose(
        time_to_longdouble(t) * u.day, ld * u.day, rtol=0, atol=1.0 * u.ns
    )


# @pytest.mark.xfail
@given(reasonable_mjd())
@example(i_f=(65536, 3.552713678800502e-15))
@example(i_f=(43143, 9.313492199680697e-10))
@example(i_f=(40000, -4.440892098500627e-16))
@example(i_f=(40000, -4.440892098500627e-16))
@pytest.mark.parametrize("scale", ["tai", "tt", "tdb"])
def test_time_from_longdouble(scale, i_f):
    i, f = i_f
    t = Time(val=i, val2=f, format="mjd", scale=scale)
    ld = np.longdouble(i) + np.longdouble(f)
    assert (
        abs(time_from_longdouble(ld, format="mjd", scale=scale) - t).to(u.ns) < 1 * u.ns
    )


@given(reasonable_mjd())
@example(i_f=(40000, 0.7333333333333333))
@example(i_f=(41498, 0.9999999999999982))
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_from_longdouble_utc(format, i_f):
    with warnings.catch_warnings():
        # Plenty of dubious years from hypothesis
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        assume(
            format != "pulsar_mjd" or i not in leap_sec_days or (1 - f) * 86400 >= 1e-9
        )
        t = Time(val=i, val2=f, format=format, scale="utc")
        ld = np.longdouble(i) + np.longdouble(f)
        assert (
            abs(time_from_longdouble(ld, format=format, scale="utc") - t).to(u.ns)
            < 1 * u.ns
        )


# time_to_mjd_string, time_from_mjd_string


# @pytest.mark.xfail
@given(reasonable_mjd())
@example(i_f=(65536, 3.552713678800502e-15))
@example(i_f=(43143, 9.313492199680697e-10))  # format="pulsar_mjd"
@example(i_f=(40000, -4.440892098500627e-16))  # format="pulsar_mjd"
@example(i_f=(40000, -4.440892098500627e-16))  # format="mjd"
@example(i_f=(40001, -4.440892098500627e-16))  # format="pulsar_mjd"
@example(i_f=(50081, 1.0000000016292463))  # format="pulsar_mjd"
@example(i_f=(43143, 9.313492199680697e-10))
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_to_longdouble_close_to_time_to_mjd_string(format, i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        t = Time(val=i, val2=f, format=format, scale="utc")
        tld = time_to_longdouble(t)
        tstr = time_to_mjd_string(t)
        # NOTE: have to add str() here, because of a numpy bug which treats
        # numpy string type differently from python str.
        # See https://github.com/numpy/numpy/issues/15608
        tld_str = np.longdouble(str(tstr))
        assert abs(tld_str - tld) * u.day < 1 * u.ns


@given(reasonable_mjd())
@example(i_f=(65536, 3.552713678800502e-15))
def test_time_to_longdouble_no_longer_than_time_to_mjd_string(i_f):
    i, f = i_f
    t = Time(val=i, val2=f, format="mjd", scale="tai")
    assert len(time_to_mjd_string(t)) >= len(str(time_to_longdouble(t)))


# @pytest.mark.xfail
@given(reasonable_mjd())
@example(i_f=(65536, 3.552713678800502e-15))
@example(i_f=(43143, 9.313492199680697e-10))  # format="pulsar_mjd"
@example(i_f=(40000, -4.440892098500627e-16))  # format="pulsar_mjd"
@example(i_f=(40000, -4.440892098500627e-16))  # format="mjd"
@example(i_f=(40001, -4.440892098500627e-16))  # format="pulsar_mjd"
@example(i_f=(43143, 9.313492199680697e-10))  # format="mjd"
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_to_mjd_string_versus_longdouble(format, i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        m = i + np.longdouble(f)
        t = Time(val=i, val2=f, format=format, scale="utc")
        tstr = time_to_mjd_string(t)
        # NOTE: have to add str() here, because of a numpy bug which treats
        # numpy string type differently from python str.
        # See https://github.com/numpy/numpy/issues/15608
        tld_str = np.longdouble(str(tstr))
        assert abs(tld_str - m) * u.day < 1 * u.ns


@given(reasonable_mjd())
@example(i_f=(65536, 3.552713678800502e-15))
@example(i_f=(43143, 9.313492199680697e-10))  # format="pulsar_mjd"
@example(i_f=(40000, -4.440892098500627e-16))  # format="pulsar_mjd"
@example(i_f=(40000, -4.440892098500627e-16))  # format="mjd"
@example(i_f=(40001, -4.440892098500627e-16))  # format="pulsar_mjd"
@example(i_f=(43143, 9.313492199680697e-10))  # format="mjd"
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_to_mjd_string_versus_decimal(format, i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        with decimal.localcontext(decimal.Context(prec=40)):
            m = Decimal(i) + Decimal(f)
            t = Time(val=i, val2=f, format=format, scale="utc")
            assert (abs(Decimal(time_to_mjd_string(t)) - m) * u.day).to(u.ns) < 1 * u.ns


@given(reasonable_mjd())
def test_time_from_mjd_string_versus_longdouble_tai(i_f):
    i, f = i_f
    m = np.longdouble(i) + np.longdouble(f)
    s = str(m)
    assert (
        abs(
            time_from_mjd_string(s, scale="tai") - time_from_longdouble(m, scale="tai")
        ).to(u.ns)
        < 1 * u.ns
    )


@given(reasonable_mjd())
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_from_mjd_string_versus_longdouble_utc(format, i_f):
    i, f = i_f
    m = np.longdouble(i) + np.longdouble(f)
    s = str(m)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        # warning is emitted during the .to() right here
        assert (
            abs(
                time_from_mjd_string(s, scale="utc", format=format)
                - time_from_longdouble(s, scale="utc", format=format)
            ).to(u.ns)
            < 1 * u.ns
        )


# pulsar_mjd


@given(reasonable_mjd())
@example(i_f=(40000, -8.881784197001254e-16))
@example(i_f=(69666, -1.4552476513551895))
def test_pulsar_mjd_never_differs_too_much_from_mjd_tai(i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        t = Time(val=i, val2=f, format="mjd", scale="tai")
        t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="tai")
        assert abs(t2 - t) <= 1 * u.ns


@given(reasonable_mjd())
@example(i_f=(40000, -8.881784197001254e-16))
@example(i_f=(43510, -1.0000000000000002))
@example(i_f=(41498, 0.7333333333333333))
def test_pulsar_mjd_never_differs_too_much_from_mjd_utc(i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        t = Time(val=i, val2=f, format="mjd", scale="utc")
        t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="utc")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
            assert abs(t2 - t).to(u.s) < (1 + 1e-10) * u.s


def test_posvel_respects_longdouble():
    pos = np.ones(3, dtype=np.longdouble)
    pos[0] += np.finfo(np.longdouble).eps
    vel = np.ones(3, dtype=np.longdouble)
    vel[1] += np.finfo(np.longdouble).eps
    pv = PosVel(pos, vel)
    assert_array_equal(pv.pos, pos)
    assert_array_equal(pv.vel, vel)


@given(reasonable_mjd())
@example(i_f=(40000, 1.2434497875801756e-14))
@example(i_f=(43875, -1.000000000000002))
@example(i_f=(48803, 1.0769154457079824e-09))
@example(i_f=(48803, 1.0000079160299438e-06))
@pytest.mark.parametrize("format", ["pulsar_mjd", "mjd"])
def test_time_from_mjd_string_accuracy_vs_longdouble(format, i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        mjd = np.longdouble(i) + np.longdouble(f)
        assume(
            format != "pulsar_mjd" or i not in leap_sec_days or (1 - f) * 86400 >= 1e-9
        )
        s = str(mjd)
        t = Time(val=i, val2=f, format=format, scale="utc")
        assert (
            abs(time_from_mjd_string(s, format=format, scale="utc") - t).to(u.us)
            < 1 * u.us
        )
        if 40000 <= i <= 60000:
            assert (
                abs(time_from_mjd_string(s, format=format, scale="utc") - t).to(u.us)
                < 1 * u.ns
            )


@pytest.mark.parametrize("format", ["pulsar_mjd", "mjd"])
def test_time_from_mjd_string_roundtrip_very_close(format):
    i = 50000
    f = np.finfo(float).eps
    t = Time(val=i, val2=f, format=format, scale="utc")
    s = time_to_mjd_string(t)
    assert abs(time_from_mjd_string(s) - t).to(u.ns) <= 4 * time_eps


@given(reasonable_mjd())
@pytest.mark.parametrize("format", ["pulsar_mjd", "mjd"])
def test_time_from_mjd_string_roundtrip(format, i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        assume(
            format != "pulsar_mjd" or i not in leap_sec_days or (1 - f) * 86400 >= 1e-9
        )
        t = Time(val=i, val2=f, format=format, scale="utc")
        assert (
            abs(t - time_from_mjd_string(time_to_mjd_string(t), format=format)).to(u.ns)
            < 1 * u.ns
        )


@given(reasonable_mjd())
def test_mjd_equals_pulsar_mjd_in_tai(i_f):
    i, f = i_f
    t = Time(val=i, val2=f, format="mjd", scale="tai")
    t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="tai")
    assert t == t2


def test_astropy_time_epsilon():
    t1 = Time(val=50000, val2=np.finfo(float).eps, format="mjd", scale="utc")
    t2 = Time(val=50000, val2=0, format="mjd", scale="utc")
    assert t1 != t2
    assert t1 - t2 > 0.9 * np.finfo(float).eps * u.day
    assert t1 - t2 <= 1.1 * np.finfo(float).eps * u.day


@pytest.mark.xfail(
    reason="ERFA doesn't like outrageous times and I haven't implemented an "
    "array-capable fallback"
)
@given(any_mjd())
@example(i_f=(-2431739, -4.440892098500627e-16))
@example(i_f=(-2431740, 0.0))
def test_make_pulsar_mjd_ancient(i_f):
    i, f = i_f
    Time(val=i, val2=f, format="pulsar_mjd", scale="utc")


@given(any_mjd())
@example(i_f=(-2431739, -4.440892098500627e-16))
@example(i_f=(-1, 0.9))
@example(i_f=(0, -0.00010000000000021106))
@example(i_f=(40000, 1.2434497875801756e-14))
@example(i_f=(524288, 1.1567635738174434e-11))
@example(i_f=(43875, -1.000000000000002))
@example(i_f=(48803, 1.0769154457079824e-09))
@example(i_f=(48803, 1.0000079160299438e-06))
def test_make_mjd_ancient(i_f):
    i, f = i_f
    Time(val=i, val2=f, format="mjd", scale="utc")


@given(normal_mjd())
@example(i_f=(41316, 3.338154197507493e-10))
@example(i_f=(40000, 0.7333333333333333))
def test_pulsar_mjd_equals_mjd_on_non_leap_second_days(i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        t1 = Time(val=i, val2=f, format="mjd", scale="utc")
        t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="utc")
        assert abs(t1 - t2).to(u.ns) < 5 * time_eps


@given(leap_sec_day_mjd())
@example(i_f=(41498, 0.7333333333333333))
def test_pulsar_mjd_equals_mjd_on_leap_second_days(i_f):
    i, f = i_f
    assume(f != 1)
    assume(f != 0)
    t1 = Time(val=i, val2=f * 86400 / 86401, format="mjd", scale="utc")
    # t1 = Time(val=i, val2=f * 86400 / 86401, format="mjd", scale="utc")
    t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="utc")
    assert abs(t1 - t2).to(u.ns) < 4 * time_eps


@given(leap_sec_day_mjd())
@example(i_f=(41498, 0.7333333333333333))
def test_pulsar_mjd_close_to_mjd_on_leap_second_days(i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        assume(f != 1)
        assume(f != 0)
        t1 = Time(val=i, val2=f, format="mjd", scale="utc")
        t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="utc")
        target((t1 - t2).to_value(u.s), label="positive difference")
        target((t2 - t1).to_value(u.s), label="negative difference")
        assert abs(t1 - t2).to(u.s) < 1.1 * u.s
        # assert abs(t1 - t2).to(u.ns) < 4 * time_eps


@given(leap_sec_day_mjd())
def test_pulsar_mjd_proceeds_at_normal_rate_on_leap_second_days(i_f):
    i, f = i_f
    assume(0.2 < f < 1)
    t1 = Time(val=i, val2=0, format="pulsar_mjd", scale="utc")
    t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="utc")
    assert abs((t2 - t1).to(u.day) / (f * u.day) - 1) < 1e-9


@given(leap_sec_day_mjd())
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_mjd_proceeds_slower_on_leap_second_days(i_f):
    i, f = i_f
    assume(0.2 < f < 1)
    t1 = Time(val=i, val2=0, format="mjd", scale="utc")
    t2 = Time(val=i, val2=f, format="mjd", scale="utc")
    assert abs((t2 - t1).to(u.s) / (f * 86401 * u.s) - 1) < 1e-9


def test_pulsar_mjd_likes_arrays():
    i = np.arange(50000, 50010)
    f = np.zeros(len(i))
    Time(val=i, val2=f, format="mjd", scale="utc")
    Time(val=i, val2=f, format="pulsar_mjd", scale="utc")


# Checking with raw ERFA


@given(leap_sec_day_mjd())
@example(i_f=(41498, 0.5000057870370372))
def test_erfa_conversion_on_leap_sec_days(i_f):
    _test_erfa_jd2cal_roundtrip(leap=True, i_f=i_f)


@given(normal_mjd())
@example(i_f=(41497, 0.5000000000000001))
@example(i_f=(41316, 9.280065826899888e-09))
@example(i_f=(40000, 0.7333333333333333))
@example(i_f=(41497, 0.7333333333333333))
@example(i_f=(41316, 9.280065826899888e-09))
def test_erfa_conversion_normal(i_f):
    _test_erfa_jd2cal_roundtrip(leap=False, i_f=i_f)


def _test_erfa_jd2cal_roundtrip(leap, i_f):
    # We need JDs not MJDs
    i_in, f_in = i_f
    assume(0 <= f_in < 1)
    if leap:
        assume(i_in in leap_sec_days)
    else:
        assume(i_in not in leap_sec_days)
    jd1_in, jd2_in = day_frac(erfa.DJM0 + i_in, f_in)

    # ERFA forward
    y, mo, d, f = erfa.jd2cal(jd1_in, jd2_in)
    assert 0 < y < 3000
    assert 0 < mo <= 12
    assert 0 <= d < 32
    assert 0 <= f < 1

    # ERFA backward - integer day
    jd1_temp, jd2_temp = erfa.cal2jd(y, mo, d)
    jd1_temp, jd2_temp = day_frac(jd1_temp, jd2_temp)  # improve numerics
    jd1_temp, jd2_temp = day_frac(jd1_temp, jd2_temp + f)
    jd_change = abs((jd1_temp - jd1_in) + (jd2_temp - jd2_in)) * u.day
    assert jd_change.to(u.ns) < 1 * u.ns

    # ERFA backward - setting up for fractional day
    # Need to tidy up the fractional day into fractional seconds
    ft = 24 * f
    h = safe_kind_conversion(np.floor(ft), dtype=int)
    ft -= h
    ft *= 60
    m = safe_kind_conversion(np.floor(ft), dtype=int)
    ft -= m
    ft *= 60
    s = ft
    assert 0 <= h < 24
    assert 0 <= m < 60
    assert 0 <= s < 60

    # ERFA backward - dtf2d including fractional day
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        jd1, jd2 = erfa.dtf2d("UTC", y, mo, d, h, m, s)
        assert jd1 == np.floor(jd1) + 0.5
        assert 0 <= jd2 < 1

    # Check whether jd1/jd2 is close to jd1_in/jd2_in
    jd1, jd2 = day_frac(jd1, jd2)
    jd_change = ((jd1 - jd1_in) + (jd2 - jd2_in)) * u.day
    target(jd_change.to_value(u.ns), label="positive difference")
    target(-jd_change.to_value(u.ns), label="negative difference")

    if leap:
        # Leap second days are a bit special
        assert abs(jd_change).to(u.s) < 1 * u.s
    else:
        # Normal days should be very close
        assert abs(jd_change).to(u.ns) < 2 * u.ns


# Try to nail down ERFA behaviour

_new_ihmsfs_dtype = np.dtype([(str(c), np.intc) for c in "hmsf"])


def d2tf_nice(ndp, days):
    sgn, hmsf = erfa.d2tf(ndp, days)
    if sgn == b"+":
        sgn = "+"
    elif sgn == b"-":
        sgn = "-"
    else:
        raise ValueError("Mysterious sign found: {!r}".format(sgn))
    if not hmsf.dtype.names:
        hmsf = hmsf.view(_new_ihmsfs_dtype)
    h = hmsf["h"]
    m = hmsf["m"]
    s = hmsf["s"] + hmsf["f"] / 10.0**ndp

    return sgn, h, m, s


def tf2d_nice(sgn, h, m, s):
    return erfa.tf2d(sgn, h, m, s)


@given(floats(-2, 2, allow_nan=False))
@example(f=1.3322676295501882e-15)  # ndp=10
@example(f=4.440892098500627e-16)  # ndp=11
@example(f=4.440892098500627e-16)  # ndp=12
@example(f=0.50390625)  # ndp=10
@pytest.mark.parametrize(
    "ndp, k",
    [
        (8, 1),
        (9, 1),
        pytest.param(10, 100, marks=pytest.mark.xfail(reason="ERFA limitations")),
        pytest.param(11, 1000, marks=pytest.mark.xfail(reason="ERFA limitations")),
        pytest.param(12, 10000, marks=pytest.mark.xfail(reason="ERFA limitations")),
    ],
)
def test_d2tf_tf2d_roundtrip(ndp, k, f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*ihour outside range", erfa.ErfaWarning)
        # Sometimes d2tf returns an hour of 24, which makes tf2d complain
        assert abs(tf2d_nice(*d2tf_nice(ndp, f)) - f) * 86400 < k * 10 ** (-ndp)


# New functions


def decimalify(i, f):
    return Decimal(i) + Decimal(f)


def assert_closer_than_ns(i_f, i_f_2, amt):
    d = decimalify(*i_f) * 86400 * 10**9
    d_2 = decimalify(*i_f_2) * 86400 * 10**9
    assert abs(d_2 - d) < amt


@given(reasonable_mjd())
def test_mjd_jd_round_trip(i_f):
    with decimal.localcontext(decimal.Context(prec=40)):
        assert_closer_than_ns(jds_to_mjds(*mjds_to_jds(*i_f)), i_f, 1)


@given(reasonable_mjd())
def test_mjd_jd_pulsar_round_trip(i_f):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*dubious year", erfa.ErfaWarning)
        i, f = i_f
        assume(i not in leap_sec_days or (1 - f) * 86400 >= 1e-9)
        with decimal.localcontext(decimal.Context(prec=40)):
            jds = mjds_to_jds_pulsar(*i_f)
            assert_closer_than_ns(jds_to_mjds_pulsar(*jds), i_f, 1)


# @pytest.mark.xfail(
#    reason="Round-trip roundoff error winds up in a leap second; probably fine"
# )
@given(leap_sec_day_mjd())
@example(i_f=(41498, 0.9999999999999982))
def test_mjd_jd_pulsar_round_trip_leap_sec_day_edge(i_f):
    with decimal.localcontext(decimal.Context(prec=40)):
        jds = mjds_to_jds_pulsar(*i_f)
        assert_closer_than_ns(jds_to_mjds_pulsar(*jds), i_f, 1)


def test_mjds_to_jds_pulsar_ok_near_leap_second():
    """
    Since we've never had a negative leap second, there are no pulsar_mjd
    times that don't correspond to any real moment. Check we got this the
    right way around.
    """
    i = 41498
    f = 86400.5 / 86401
    mjds_to_jds_pulsar(i, f)


def test_jds_to_mjds_pulsar_raises_during_leap_second():
    i = 41498
    f = 86400.5 / 86401
    jd1, jd2 = mjds_to_jds(i, f)
    with pytest.raises(ValueError):
        jds_to_mjds_pulsar(jd1, jd2)


@given(reasonable_mjd())
@example(i_f=(41498, 0.9999999999999982))
@example(i_f=(40000, 1.2434497875801756e-14))
@example(i_f=(43875, -1.000000000000002))
@example(i_f=(48803, 1.0769154457079824e-09))
@example(i_f=(48803, 1.0000079160299438e-06))
@example(i_f=(43143, 9.313492199680697e-10))
def test_str_to_mjds(i_f):
    i, f = i_f
    with decimal.localcontext(decimal.Context(prec=40)):
        assert_closer_than_ns(str_to_mjds(str(decimalify(i, f))), i_f, 1)


@given(reasonable_mjd())
@example(i_f=(43143, 9.313492199680697e-10))
@example(i_f=(41498, 0.9999999999999982))
def test_mjds_to_str(i_f):
    i, f = i_f
    with decimal.localcontext(decimal.Context(prec=40)):
        s = mjds_to_str(i, f)
        d = Decimal(s) * 86400 * 10**9
        d2 = decimalify(i, f) * 86400 * 10**9
        assert abs(d2 - d) < 1


@given(reasonable_mjd())
@example(i_f=(43143, 9.313492199680697e-10))
# @settings(max_examples=5000)
def test_mjds_to_str_roundtrip(i_f):
    i, f = i_f
    d = (decimalify(i, f) * u.day).to(u.ns)
    i_o, f_o = str_to_mjds(mjds_to_str(i, f))
    d_o = (decimalify(i_o, f_o) * u.day).to(u.ns)
    assert abs(d_o - d) < time_eps


@given(reasonable_mjd())
@example(i_f=(65536, 3.637978807091714e-12))
def test_day_frac(i_f):
    assert_closer_than_ns(day_frac(*i_f), i_f, 1)


@given(reasonable_mjd())
@example(i_f=(65536, 3.637978807091714e-12))
def test_two_sum(i_f):
    assert_closer_than_ns(two_sum(*i_f), i_f, 1)
