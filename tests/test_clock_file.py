from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from numpy.testing import assert_allclose, assert_array_equal

from pint.observatory import get_observatory
from pint.observatory.clock_file import (
    ClockFile,
    ConstructedClockFile,
    write_tempo2_clock_file,
    write_tempo_clock_file,
)


def t(mjd):
    return Time(mjd, format="pulsar_mjd", scale="utc")


@pytest.fixture
def basic_clock():
    return ConstructedClockFile(
        mjd=np.array([50000, 55000, 60000]), clock=np.array([1.0, 2.0, -1.0]) * u.us
    )


def test_merge_clocks_values():
    a = np.array([50000, 60000])
    av = np.array([0, 2]) * u.us
    b = np.array([50000, 55000, 60000])
    bv = np.array([0, 0, 1]) * u.us

    ca = ConstructedClockFile(mjd=a, clock=av)
    cb = ConstructedClockFile(mjd=b, clock=bv)

    m = ClockFile.merge([ca, cb])

    ts = t(np.linspace(50000, 60000, 10))
    assert_allclose(
        m.evaluate(ts).to_value(u.us),
        ca.evaluate(ts).to_value(u.us) + cb.evaluate(ts).to_value(u.us),
    )


def test_merge_clocks_values_repeat():
    a = np.array([50000, 60000])
    av = np.array([0, 2]) * u.us
    b = np.array([50000, 55000, 55000, 60000])
    bv = np.array([0, 0, 1, 1]) * u.us

    ca = ConstructedClockFile(mjd=a, clock=av)
    cb = ConstructedClockFile(mjd=b, clock=bv)

    m = ClockFile.merge([ca, cb])

    ts = t(np.linspace(50000, 60000, 10))
    assert_allclose(
        m.evaluate(ts).to_value(u.us),
        ca.evaluate(ts).to_value(u.us) + cb.evaluate(ts).to_value(u.us),
    )


def test_merge_clocks_values_repeat_more():
    a = np.array([50000, 60000])
    av = np.array([0, 2]) * u.us
    b = np.array([50000, 55000, 55000, 55000, 55000, 60000])
    bv = np.array([0, 0, 7, 8, 1, 1]) * u.us

    ca = ConstructedClockFile(mjd=a, clock=av)
    cb = ConstructedClockFile(mjd=b, clock=bv)

    m = ClockFile.merge([ca, cb])

    ts = t(np.linspace(50000, 60000, 10))
    assert_allclose(
        m.evaluate(ts).to_value(u.us),
        ca.evaluate(ts).to_value(u.us) + cb.evaluate(ts).to_value(u.us),
    )


def test_merge_clocks_preserves_discontinuities():
    a = np.array([50000, 60000])
    av = np.array([2, 2]) * u.us
    b = np.array([50000, 55000, 55000, 60000])
    bv = np.array([0, 0, 1, 1]) * u.us

    ca = ConstructedClockFile(mjd=a, clock=av)
    cb = ConstructedClockFile(mjd=b, clock=bv)

    m = ClockFile.merge([ca, cb])

    assert m.evaluate(t(54999)) == m.evaluate(t(50000))
    assert m.evaluate(t(55001)) == m.evaluate(t(60000))


def test_merge_mjds_trims_range():
    a = np.array([50000, 60000])
    b = np.array([40000, 55000, 61000])

    ca = ConstructedClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ConstructedClockFile(mjd=b, clock=np.zeros_like(b) * u.s)

    m = ClockFile.merge([ca, cb])
    assert_array_equal(m.time.mjd, np.array([50000, 55000, 60000]))


def test_merge_mjds_trims_range_repeat_beginning():
    a = np.array([50000, 50000, 60000])
    b = np.array([40000, 55000, 61000])

    ca = ConstructedClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ConstructedClockFile(mjd=b, clock=np.zeros_like(b) * u.s)

    m = ClockFile.merge([ca, cb])
    assert_array_equal(m.time.mjd, np.array([50000, 50000, 55000, 60000]))


def test_merge_mjds_trims_range_repeat_end():
    a = np.array([50000, 60000, 60000])
    b = np.array([40000, 55000, 61000])

    ca = ConstructedClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ConstructedClockFile(mjd=b, clock=np.zeros_like(b) * u.s)

    m = ClockFile.merge([ca, cb])
    assert_array_equal(m.time.mjd, np.array([50000, 55000, 60000, 60000]))


def test_merge_mjds_trims_range_mixed():
    a = np.array([50000, 61000])
    b = np.array([40000, 55000, 60000])

    ca = ConstructedClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ConstructedClockFile(mjd=b, clock=np.zeros_like(b) * u.s)
    m = ClockFile.merge([ca, cb])
    assert_array_equal(m.time.mjd, np.array([50000, 55000, 60000]))


def test_tempo2_round_trip(basic_clock):
    hdrline = "# fake conversion test"
    f = StringIO()
    write_tempo2_clock_file(f, hdrline=hdrline, clock=basic_clock)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo2")

    assert_array_equal(read_clock.time.mjd, basic_clock.time.mjd)
    assert_array_equal(
        read_clock.clock.to_value(u.us), basic_clock.clock.to_value(u.us)
    )


def test_tempo2_round_trip_arecibo():
    ao = get_observatory("arecibo")
    ao.last_clock_correction_mjd()
    clock = ao._clock[0]

    hdrline = "# fake conversion test"

    f = StringIO()
    write_tempo2_clock_file(f, hdrline=hdrline, clock=clock)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo2")

    assert_allclose(read_clock.time.mjd, clock.time.mjd)
    assert_allclose(read_clock.clock.to_value(u.us), clock.clock.to_value(u.us))


def test_tempo_round_trip(basic_clock):
    obscode = "["
    f = StringIO()
    write_tempo_clock_file(f, obscode=obscode, clock=basic_clock)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo")

    assert_array_equal(read_clock.time.mjd, basic_clock.time.mjd)
    assert_array_equal(
        read_clock.clock.to_value(u.us), basic_clock.clock.to_value(u.us)
    )


def test_tempo_round_trip_arecibo():
    ao = get_observatory("arecibo")
    ao.last_clock_correction_mjd()
    clock = ao._clock[0]

    obscode = "1"

    f = StringIO()
    write_tempo_clock_file(f, obscode=obscode, clock=clock)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo")

    assert_allclose(read_clock.time.mjd, clock.time.mjd)
    assert_allclose(read_clock.clock.to_value(u.us), clock.clock.to_value(u.us))
