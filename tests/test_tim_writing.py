from io import StringIO
import re

import astropy.units as u
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import lists, tuples, from_regex
from pint.toa import get_TOAs

basic_tim_header = "FORMAT 1\n"

basic_tim = """
53358.000056.3.000.000.9y.x.ff 424.000000 53358.767912764015642   1.277  ao  -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 903.18 -tmplt B1855+09.430.PUPPI.9y.x.sum.sm -gof 1.09 -nbin 2048 -nch 1 -chan 1 -subint 0 -snr 142.71 -wt 15 -proc 9y -pta NANOGrav -to -0.789e-6
54862.000023.3.000.000.9y.x.ff 428.000000 54862.593840761061363   1.337  ao  -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 903.18 -tmplt B1855+09.430.PUPPI.9y.x.sum.sm -gof 0.921 -nbin 2048 -nch 1 -chan 2 -subint 1 -snr 131.03 -wt 15 -proc 9y -pta NANOGrav -to -0.789e-6
53453.000000.3.000.000.9y.x.ff 1434.000000 53453.471481198627050   0.433  ao  -fe L-wide -be ASP -f L-wide_ASP -bw 4 -tobs 903.79 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.08 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 414.47 -wt 15 -proc 9y -pta NANOGrav -to -0.839e-6
55298.000029.3.000.000.9y.x.ff 1406.000000 55298.405393153214034   0.701  ao  -fe L-wide -be ASP -f L-wide_ASP -bw 4 -tobs 903.79 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.18 -nbin 2048 -nch 1 -chan 9 -subint 0 -snr 257.6 -wt 15 -proc 9y -pta NANOGrav -to -0.839e-6
puppi_56212_1855+09_1056.9y.x.ff 1511.510010 56212.923999620852331   0.679  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1527.2 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.04 -nbin 2048 -nch 8 -chan 21 -subint 2 -snr 266.29 -wt 14317 -proc 9y -pta NANOGrav
puppi_56374_1855+09_0482.9y.x.ff 1224.500000 56374.458848469919109   1.865  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1229.9 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 0.966 -nbin 2048 -nch 8 -chan 44 -subint 1 -snr 96.158 -wt 15166 -proc 9y -pta NANOGrav
puppi_56577_B1855+09_0547.9y.x.ff 1687.030029 56577.922801352174762   0.284  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 2.92 -nbin 2048 -nch 8 -chan 7 -subint 0 -snr 619.21 -wt 15011 -proc 9y -pta NANOGrav
puppi_56598_B1855+09_0089.9y.x.ff 1174.531006 56598.871995326598351   1.110  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.13 -nbin 2048 -nch 8 -chan 48 -subint 0 -snr 167.54 -wt 15011 -proc 9y -pta NANOGrav
puppi_56598_B1855+09_0089.9y.x.ff 1153.437012 56598.871995343722109   2.613  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.22 -nbin 2048 -nch 8 -chan 50 -subint 0 -snr 72.635 -wt 5629 -proc 9y -pta NANOGrav
"""


def do_roundtrip(toas, format="tempo2", **kwargs):
    f = StringIO()
    toas.write_TOA_file(f, format=format)
    toas_2 = get_TOAs(StringIO(f.getvalue()), **kwargs)
    # assert toas.commands == toas_2.commands
    assert toas.ntoas == toas_2.ntoas
    assert all(
        abs(x) < 1 * u.ns
        for x in (
            toas.get_mjds(high_precision=True) - toas_2.get_mjds(high_precision=True)
        )
    )
    assert np.all(toas.get_freqs() == toas_2.get_freqs())
    assert (
        abs(toas.get_errors() - toas_2.get_errors()).max() < 1 * u.ns
    )  # this is weirdly large because we don't record many decimal places
    assert np.all(toas.get_obss() == toas_2.get_obss())
    assert np.all(toas.get_pulse_numbers() == toas_2.get_pulse_numbers())
    assert np.all(toas.get_flags() == toas_2.get_flags())
    return toas_2


def test_basic():
    f = StringIO(basic_tim_header + basic_tim)
    do_roundtrip(get_TOAs(f))


@pytest.mark.parametrize(
    "c",
    [
        "TIME 1",
        "EFAC 3.1415926535",
        "EQUAD 17",
        "EMIN 1",
        "EMAX 10",
        "FMIN 1500",
        "FMAX 1500",
    ],
)
def test_time(c):
    f = StringIO(f"{basic_tim_header}\n{c}\n{basic_tim}")
    do_roundtrip(get_TOAs(f))


@pytest.mark.parametrize(
    "k",
    [
        dict(ephem="DE430"),
        # dict(tdb_method="ephemeris"),
        dict(planets=True),
        dict(include_gps=False),
        dict(include_bipm=False),
        dict(bipm_version="BIPM2019"),
        dict(bipm_version="BIPM2003"),
    ],
)
def test_options(k):
    f = StringIO(basic_tim_header + basic_tim)
    do_roundtrip(get_TOAs(f, **k), **k)


@pytest.mark.parametrize(
    "k",
    [
        dict(ephem="DE430"),
        # dict(tdb_method="ephemeris"),
        dict(planets=True),
        dict(include_gps=False),
        dict(include_bipm=False),
        dict(bipm_version="BIPM2019"),
        dict(bipm_version="BIPM2003"),
    ],
)
def test_barycenter(k):
    f = StringIO(
        basic_tim_header
        + """
some_barycentered 999999.999 56400.000000000000000   1.000  @  -some argument -another argument
some_barycentered 999999.999 56401.000000000000000   1.000  @  -some argument -another argument
some_barycentered 999999.999 56402.000000000000000   1.000  @  -some argument -another argument
some_barycentered 999999.999 56403.000000000000000   1.000  @  -some argument -another argument
"""
        + basic_tim
    )
    do_roundtrip(get_TOAs(f, **k), **k)


@given(
    lists(
        tuples(
            from_regex(re.compile(r"[ \t]+", re.ASCII), fullmatch=True),
            from_regex(re.compile(r"-[a-zA-Z_]\w+", re.ASCII), fullmatch=True).filter(
                lambda t: t.lower()
                not in {
                    "-error",
                    "-freq",
                    "-scale",
                    "-mjd",
                    "-flags",
                    "-obs",
                    "-clkcorr",
                    "-to",  # FIXME: used in clock corrections? What does this do?
                    "-pn",  # Gets converted to pulse_number so fails roundtrip
                }
            ),
            from_regex(re.compile(r"[ \t]+", re.ASCII), fullmatch=True),
            from_regex(re.compile(r"\w+", re.ASCII), fullmatch=True),
        ).map(lambda t: "".join(t))
    ).map(lambda t: "".join(t))
)
def test_flags(s):
    s = "\n".join(
        [
            basic_tim_header,
            f"""some_barycentered 999999.999 56400.000000000000000   1.000  @{s}""",
            basic_tim,
        ]
    )
    f = StringIO(s)
    toas = get_TOAs(f)
    do_roundtrip(toas)


def test_pulse_number():
    t = get_TOAs(StringIO(basic_tim_header + basic_tim))
    t.table["pulse_number"] = np.random.randint(10**9, size=t.ntoas)
    do_roundtrip(t)


def test_name_preservation():
    f = StringIO(basic_tim_header + basic_tim)
    t = get_TOAs(f)
    v, _ = t.get_flag_value("name")
    v = set(v)
    assert "53453.000000.3.000.000.9y.x.ff" in v
    assert "puppi_56577_B1855+09_0547.9y.x.ff" in v
    t2 = do_roundtrip(t)
    v, _ = t2.get_flag_value("name")
    v = set(v)
    assert "53453.000000.3.000.000.9y.x.ff" in v
    assert "puppi_56577_B1855+09_0547.9y.x.ff" in v
