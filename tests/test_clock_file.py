from io import StringIO
from textwrap import dedent

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from numpy.testing import assert_allclose, assert_array_equal

from pint.observatory import get_observatory
from pint.observatory.clock_file import (
    ClockFile,
    ConstructedClockFile,
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
    hdrline = "# FAKE1 FAKE2"
    f = StringIO()
    basic_clock.write_tempo2_clock_file(f, hdrline=hdrline)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo2")

    assert_allclose(read_clock.time.mjd, basic_clock.time.mjd)
    assert_allclose(read_clock.clock.to_value(u.us), basic_clock.clock.to_value(u.us))


def test_tempo2_round_trip_file(basic_clock, tmp_path):
    f = tmp_path / "fake.clk"
    hdrline = "# FAKE1 FAKE2"
    basic_clock.write_tempo2_clock_file(str(f), hdrline=hdrline)
    read_clock = ClockFile.read(str(f), format="tempo2")

    assert_allclose(read_clock.time.mjd, basic_clock.time.mjd)
    assert_allclose(read_clock.clock.to_value(u.us), basic_clock.clock.to_value(u.us))


def test_tempo_round_trip_file(basic_clock, tmp_path):
    f = tmp_path / "fake.clk"
    obscode = "{"
    basic_clock.write_tempo_clock_file(str(f), obscode=obscode)
    read_clock = ClockFile.read(str(f), format="tempo")

    assert_allclose(read_clock.time.mjd, basic_clock.time.mjd)
    assert_allclose(read_clock.clock.to_value(u.us), basic_clock.clock.to_value(u.us))


loadable_observatories = ["gbt", "arecibo", "fast", "gb140", "gb853", "jb", "wsrt"]


@pytest.mark.parametrize("obs", loadable_observatories)
def ensure_can_read(obs):
    o = get_observatory(obs)
    o.last_clock_correction_mjd()


def test_tempo2_round_trip_arecibo():
    ao = get_observatory("arecibo")
    ao.last_clock_correction_mjd()
    clock = ao._clock[0]

    hdrline = "# fake conversion test"

    f = StringIO()
    clock.write_tempo2_clock_file(f, hdrline=hdrline)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo2")

    assert_allclose(read_clock.time.mjd, clock.time.mjd)
    assert_allclose(read_clock.clock.to_value(u.us), clock.clock.to_value(u.us))


def test_tempo_round_trip(basic_clock):
    obscode = "["
    f = StringIO()
    basic_clock.write_tempo_clock_file(f, obscode=obscode)
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
    clock.write_tempo_clock_file(f, obscode=obscode)
    read_clock = ClockFile.read(StringIO(f.getvalue()), format="tempo")

    assert_allclose(read_clock.time.mjd, clock.time.mjd)
    assert_allclose(read_clock.clock.to_value(u.us), clock.clock.to_value(u.us))


def test_tempo2_round_trip_comments():
    contents = dedent(
        """\
        # FAKE1 FAKE2 3 header comments
        # Initial comments
        # covering several lines
        50000.00000 0.000001000000 And some text
        50001.00000 0.000002000000
        # A commenty line
        50002.00000 0.000003000000 same-line text
        # and a commenty line
        """
    )
    c = ClockFile.read(StringIO(contents), format="tempo2")
    o = StringIO()
    c.write_tempo2_clock_file(o)
    assert o.getvalue() == contents


def test_tempo_round_trip_comments():
    contents = dedent(
        """\
           MJD       EECO-REF    NIST-REF NS      DATE    COMMENTS
        =========    ========    ======== ==    ========  ========
        # Initial comments
        # covering several lines
         50000.00       0.000       1.000 1    10-Oct-95  And some text
         50001.00       0.000       1.000 1    11-Oct-95
        # A commenty line
         50002.00       0.000       1.000 1    12-Oct-95  same-line text
        # and a commenty line
        """
    )
    c = ClockFile.read(StringIO(contents), format="tempo")
    o = StringIO()
    c.write_tempo_clock_file(o, obscode="1")
    assert o.getvalue() == contents


def test_leading_comment_tempo2():
    c = ClockFile.read(
        StringIO(
            dedent(
                """\
        # FAKE1 FAKE2 3 c header comments
        # Initial comments from c
        50000.00000 0.000001000000 And some text
        50001.00000 0.000002000000
        """
            )
        ),
        format="tempo2",
    )
    assert c.leading_comment == "# Initial comments from c"


def test_merge_comments():
    c1 = ClockFile.read(
        StringIO(
            dedent(
                """\
        # FAKE1 FAKE2 3 c1 header comments
        # Initial comments from c1
        # covering several lines
        50000.00000 0.000001000000 And some text
        50001.00000 0.000002000000
        50002.00000 0.000002000000
        # A commenty line
        50003.00000 0.000003000000 same-line text
        # and a commenty line
        60000.00000 0.000000000000
        """
            )
        ),
        format="tempo2",
    )
    c2 = ClockFile.read(
        StringIO(
            dedent(
                """\
        # FAKE2 FAKE3 3 c2 header comments
        # Initial comments from c2
        # covering several lines
        50000.00000 0.000001000000 From c2
        50001.00000 0.000002000000
        50002.00000 0.000002000000
        55000.00000 0.000003000000 The beginning of a jump
        55000.00000 0.000000000000 The end of a jump
        60000.00000 0.000000000000
        """
            )
        ),
        format="tempo2",
    )

    m = ClockFile.merge([c1, c2])
    o = StringIO()
    m.write_tempo2_clock_file(o, hdrline="# FAKE1 FAKE3")
    contents = dedent(
        """\
        # FAKE1 FAKE3
        # Initial comments from c1
        # covering several lines
        # Initial comments from c2
        # covering several lines
        50000.00000 0.000002000000 And some text
        # From c2
        50001.00000 0.000004000000
        50002.00000 0.000004000000
        # A commenty line
        50003.00000 0.000005000200 same-line text
        # and a commenty line
        55000.00000 0.000004500450 The beginning of a jump
        55000.00000 0.000001500450 The end of a jump
        60000.00000 0.000000000000
        """
    )
    assert o.getvalue() == contents
