import warnings
from io import StringIO
from textwrap import dedent

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from numpy.testing import assert_allclose, assert_array_equal

from pint.observatory import (
    ClockCorrectionOutOfRange,
    bipm_default,
    get_observatory,
    update_clock_files,
)
from pint.observatory.clock_file import ClockFile
from pint.observatory.topo_obs import export_all_clock_files


def t(mjd):
    return Time(mjd, format="pulsar_mjd", scale="utc")


@pytest.fixture
def basic_clock():
    return ClockFile(
        mjd=np.array([50000, 55000, 60000]),
        clock=np.array([1.0, 2.0, -1.0]) * u.us,
        friendly_name="basic_clock",
    )


def test_merge_clocks_values():
    a = np.array([50000, 60000])
    av = np.array([0, 2]) * u.us
    b = np.array([50000, 55000, 60000])
    bv = np.array([0, 0, 1]) * u.us

    ca = ClockFile(mjd=a, clock=av)
    cb = ClockFile(mjd=b, clock=bv)

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

    ca = ClockFile(mjd=a, clock=av)
    cb = ClockFile(mjd=b, clock=bv)

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

    ca = ClockFile(mjd=a, clock=av)
    cb = ClockFile(mjd=b, clock=bv)

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

    ca = ClockFile(mjd=a, clock=av)
    cb = ClockFile(mjd=b, clock=bv)

    m = ClockFile.merge([ca, cb])

    assert m.evaluate(t(54999)) == m.evaluate(t(50000))
    assert m.evaluate(t(55001)) == m.evaluate(t(60000))


def test_merge_mjds_trims_range():
    a = np.array([50000, 60000])
    b = np.array([40000, 55000, 61000])

    ca = ClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ClockFile(mjd=b, clock=np.zeros_like(b) * u.s)

    with pytest.warns(UserWarning, match="out of range"):
        m = ClockFile.merge([ca, cb])
    assert_array_equal(m.time.mjd, np.array([50000, 55000, 60000]))


def test_merge_mjds_trims_range_repeat_beginning():
    a = np.array([50000, 50000, 60000])
    b = np.array([40000, 55000, 61000])

    ca = ClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ClockFile(mjd=b, clock=np.zeros_like(b) * u.s)

    with pytest.warns(UserWarning, match="out of range"):
        m = ClockFile.merge([ca, cb])
    assert_array_equal(m.time.mjd, np.array([50000, 50000, 55000, 60000]))


def test_merge_mjds_trims_range_repeat_end():
    a = np.array([50000, 60000, 60000])
    b = np.array([40000, 55000, 61000])

    ca = ClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ClockFile(mjd=b, clock=np.zeros_like(b) * u.s)

    with pytest.warns(UserWarning, match="out of range"):
        m = ClockFile.merge([ca, cb])
    assert_array_equal(m.time.mjd, np.array([50000, 55000, 60000, 60000]))


def test_merge_mjds_trims_range_mixed():
    a = np.array([50000, 61000])
    b = np.array([40000, 55000, 60000])

    ca = ClockFile(mjd=a, clock=np.zeros_like(a) * u.s)
    cb = ClockFile(mjd=b, clock=np.zeros_like(b) * u.s)
    with pytest.warns(UserWarning, match="out of range"):
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
def test_can_read(obs):
    o = get_observatory(obs)
    o.last_clock_correction_mjd()


@pytest.mark.parametrize(
    "obs,mjd",
    [
        # This is all the observatories PINT knows about clock corrections for
        # And a day towards the end of what's in the global repository
        ("gbt", 59690),
        ("ao", 59070),
        ("fast", 58740),
        ("vla", 59270),
        ("meerkat", 59260),
        ("parkes", 58690),
        ("jodrell", 59520),
        ("effelsberg", 57190),
        ("wsrt", 57200),
        ("most", 58360),
        ("gb140", 51390),
        ("gb853", 50680),
    ],
)
def test_corrections_cover(obs, mjd):
    o = get_observatory(obs)
    o.clock_corrections(Time(mjd, format="pulsar_mjd"), limits="error")


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


def test_leading_comment_tempo():
    c = ClockFile.read(
        StringIO(
            dedent(
                """\
           MJD       EECO-REF    NIST-REF NS      DATE    COMMENTS
        =========    ========    ======== ==    ========  ========
        # Initial comments
         50000.00       0.000       1.000 1    10-Oct-95  And some text
         50001.00       0.000       1.000 1    11-Oct-95
        """
            )
        ),
        format="tempo",
    )
    assert c.leading_comment == "# Initial comments"


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
        friendly_name="c1",
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
        friendly_name="c2",
    )

    m = ClockFile.merge([c1, c2])
    o = StringIO()
    m.write_tempo2_clock_file(o, hdrline="# FAKE1 FAKE3")
    contents = dedent(
        """\
        # FAKE1 FAKE3
        # Merged from ['c1', 'c2']
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


def test_export_clock_file(tmp_path):
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
    clock_path = tmp_path / "file.clk"
    clock_path.write_text(contents)
    target_dir = tmp_path / "a"
    target_dir.mkdir()
    target_file = target_dir / "other-file.clk"
    c = ClockFile.read(clock_path, format="tempo2")
    c.export(target_file)
    assert target_file.read_text() == contents


def test_export_all_gbt(tmp_path):
    o = get_observatory("gbt")
    o.last_clock_correction_mjd()
    export_all_clock_files(tmp_path)
    assert (tmp_path / "time_gbt.dat").exists()
    assert (tmp_path / "gps2utc.clk").exists()
    assert (tmp_path / f"tai2tt_{bipm_default.lower()}.clk").exists()

    # I'd like to check that this isn't exported unless it's been used
    # but it's cached in the global TopoObs object (separate from the
    # Astropy cache)
    # assert not (tmp_path / "time_ao.dat").exists()


def test_update_clock_files_str(tmp_path):
    export_all_clock_files(str(tmp_path))


def test_update_clock_files(tmp_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*dubious year.*")
        update_clock_files()
    export_all_clock_files(tmp_path)
    assert (tmp_path / "wsrt2gps.clk").exists()


def test_out_of_range_raises_exception(basic_clock):
    with pytest.raises(ClockCorrectionOutOfRange) as excinfo:
        basic_clock.evaluate(Time(60001, format="mjd"), limits="error")


def test_out_of_range_message_has_helpful_name(basic_clock):
    with pytest.raises(ClockCorrectionOutOfRange) as excinfo:
        basic_clock.evaluate(Time(60001, format="mjd"), limits="error")
    assert "basic_clock" in str(excinfo.value)


def test_out_of_range_emits_warning(basic_clock):
    with pytest.warns(UserWarning, match="basic_clock"):
        basic_clock.evaluate(Time(60001, format="mjd"))


def test_out_of_range_allowed():
    basic_clock = ClockFile(
        mjd=np.array([50000, 55000, 60000]),
        clock=np.array([1.0, 2.0, -1.0]) * u.us,
        friendly_name="basic_clock",
        valid_beyond_ends=True,
    )
    basic_clock.evaluate(Time(60001, format="mjd"), limits="error")


def test_out_of_order_raises_exception():
    with pytest.raises(ValueError) as excinfo:
        ClockFile(
            mjd=np.array([50000, 55000, 54000, 60000]),
            clock=np.array([1.0, 2.0, -1.0, 1.0]) * u.us,
            friendly_name="basic_clock",
        )
    assert "55000" in str(excinfo.value)
