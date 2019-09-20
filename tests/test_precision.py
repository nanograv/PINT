import re
import sys

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from hypothesis import example, given, settings
from hypothesis.strategies import floats, integers, just, one_of
from numpy.testing import assert_array_equal

import pint.pulsar_mjd
from pint.utils import (
    data2longdouble,
    longdouble2str,
    str2longdouble,
    time_to_longdouble,
    time_to_mjd_string,
)


@given(
    one_of(integers(40000, 60000), integers(-3000000, 3000000), just(0)),
    floats(-2, 2, allow_nan=False),
)
def test_str_roundtrip_is_exact(i, f):
    ld = np.longdouble(i) + np.longdouble(f)
    assert ld == np.longdouble(str(ld))


@given(
    one_of(integers(40000, 60000), integers(-3000000, 3000000), just(0)),
    floats(-2, 2, allow_nan=False),
)
def test_longdouble_str_roundtrip_is_exact(i, f):
    ld = np.longdouble(i) + np.longdouble(f)
    assert ld == str2longdouble(longdouble2str(ld))


@given(
    one_of(integers(40000, 60000), integers(-3000000, 3000000), just(0)),
    one_of(floats(-2, 2, allow_nan=False), integers(-2, 2), just(0)),
)
def test_longdouble2str_same_as_str_and_repr(i, f):
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


@given(one_of(integers(40000, 70000)), floats(-2.0, 2.0, allow_nan=False))
@example(i=65536, f=3.552713678800502e-15)
@example(format="pulsar_mjd", i=43143, f=9.313492199680697e-10)
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_to_longdouble(format, i, f):
    t = Time(val=i, val2=f, format=format, scale="tai")
    ld = np.longdouble(i) + np.longdouble(f)
    assert_quantity_allclose(
        time_to_longdouble(t) * u.day, ld * u.day, rtol=0, atol=1.0 * u.ns
    )


@pytest.mark.xfail(reason="What on Earth? A day? Is this a failure of pulsar_mjd?")
@given(integers(40000, 70000), floats(-2.0, 2.0, allow_nan=False))
@example(i=65536, f=3.552713678800502e-15)
@example(format="pulsar_mjd", i=43143, f=9.313492199680697e-10)
@example(format="pulsar_mjd", i=40000, f=-4.440892098500627e-16)
@example(format="mjd", i=40000, f=-4.440892098500627e-16)
@settings(max_examples=5000)
@pytest.mark.parametrize("format", ["mjd", pytest.param("pulsar_mjd")])
def test_time_to_mjd_string_precision(format, i, f):
    t = Time(val=i, val2=f, format=format, scale="tai")
    ld = np.longdouble(i) + np.longdouble(f)
    assert_quantity_allclose(
        np.longdouble(time_to_mjd_string(t)) * u.day,
        ld * u.day,
        rtol=0,
        atol=2.0 * u.ns,
    )


@given(integers(40000, 70000), floats(-2.0, 2.0, allow_nan=False))
@example(i=65536, f=3.552713678800502e-15)
@example(format="pulsar_mjd", i=43143, f=9.313492199680697e-10)
@example(format='pulsar_mjd', i=50081, f=1.0000000016292463)
@pytest.mark.parametrize("format", ["mjd", "pulsar_mjd"])
def test_time_to_longdouble_close_to_time_to_mjd_string(format, i, f):
    t = Time(val=i, val2=f, format=format, scale="tai")
    assert_quantity_allclose(
        np.longdouble(time_to_mjd_string(t)) * u.day,
        time_to_longdouble(t) * u.day,
        rtol=0,
        atol=3.0 * u.ns,
    )


@given(integers(40000, 70000), floats(-2.0, 2.0, allow_nan=False))
@example(i=65536, f=3.552713678800502e-15)
def test_time_to_longdouble_no_longer_than_time_to_mjd_string(i, f):
    t = Time(val=i, val2=f, format="mjd", scale="tai")
    assert len(time_to_mjd_string(t)) >= len(str(time_to_longdouble(t)))


@pytest.mark.xfail(reason="What on Earth? A day? Is this a failure of pulsar_mjd?")
@given(integers(40000, 70000), floats(-2.0, 2.0, allow_nan=False))
@example(i=65536, f=3.552713678800502e-15)
@example(i=40000, f=-8.881784197001254e-16)
def test_time_to_mjd_string_format_dependence(i, f):
    t = Time(val=i, val2=f, format="mjd", scale="tai")
    t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="tai")
    assert_quantity_allclose(
        np.longdouble(time_to_mjd_string(t)) * u.day,
        np.longdouble(time_to_mjd_string(t2)) * u.day,
        rtol=0,
        atol=1.0 * u.ns,
    )


@given(integers(40000, 70000), floats(-2.0, 2.0, allow_nan=False))
@example(i=40000, f=-8.881784197001254e-16)
@settings(max_examples=5000)
def test_pulsar_mjd_never_differs_too_much_from_mjd(i, f):
    t = Time(val=i, val2=f, format="mjd", scale="tai")
    t2 = Time(val=i, val2=f, format="pulsar_mjd", scale="tai")
    assert abs(t2 - t) <= 1 * u.ns
