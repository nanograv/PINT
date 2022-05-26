from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pint.observatory.clock_file import (
    ClockFile,
    ConstructedClockFile,
    merged_mjds,
    write_tempo2_clock_file,
)


@pytest.fixture
def basic_clock():
    return ConstructedClockFile(
        mjd=np.array([50000, 55000, 60000]), clock=np.array([1.0, 2.0, -1.0]) * u.us
    )


def test_merge_mjds_removes_overlap():
    a = np.array([50000, 60000])
    b = np.array([50000, 55000, 60000])

    ca = ConstructedClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ConstructedClockFile(mjd=b, clock=np.zeros_like(b) * u.s)
    assert_array_equal(merged_mjds([ca, cb]), b)


def test_merge_mjds_trims_range():
    a = np.array([50000, 60000])
    b = np.array([40000, 55000, 61000])

    ca = ConstructedClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ConstructedClockFile(mjd=b, clock=np.zeros_like(b) * u.s)
    assert_array_equal(merged_mjds([ca, cb]), np.array([50000, 55000, 60000]))


def test_merge_mjds_trims_range_mixed():
    a = np.array([50000, 61000])
    b = np.array([40000, 55000, 60000])

    ca = ConstructedClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ConstructedClockFile(mjd=b, clock=np.zeros_like(b) * u.s)
    assert_array_equal(merged_mjds([ca, cb]), np.array([50000, 55000, 60000]))


def test_tempo2_round_trip(basic_clock):
    hdrline = "# fake conversion test"
    f = StringIO()
    write_tempo2_clock_file(f, hdrline=hdrline, clocks=basic_clock)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo2")

    assert_array_equal(read_clock.time.mjd, basic_clock.time.mjd)
    assert_array_equal(
        read_clock.clock.to_value(u.us), basic_clock.clock.to_value(u.us)
    )
