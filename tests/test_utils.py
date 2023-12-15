"""Test basic functionality of the :module:`pint.utils`."""
import io
import os
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import astropy.units as u
import numpy as np
import pytest
import scipy.stats
from astropy.time import Time
from hypothesis import assume, example, given
from hypothesis.extra.numpy import array_shapes, arrays, scalar_dtypes
from hypothesis.strategies import (
    binary,
    composite,
    floats,
    integers,
    just,
    one_of,
    sampled_from,
    slices,
    tuples,
)
from numdifftools import Derivative
from numpy.testing import assert_allclose, assert_array_equal
from pinttestdata import datadir

import pint.models as tm
from pint import fitter, toa, dmu
from pint.pulsar_mjd import (
    jds_to_mjds,
    jds_to_mjds_pulsar,
    mjds_to_jds,
    mjds_to_jds_pulsar,
    mjds_to_str,
    str_to_mjds,
    time_from_mjd_string,
    time_to_longdouble,
)
from pint.utils import (
    FTest,
    PosVel,
    compute_hash,
    dmxparse,
    interesting_lines,
    lines_of,
    list_parameters,
    open_or_use,
    taylor_horner,
    taylor_horner_deriv,
    find_prefix_bytime,
    merge_dmx,
    convert_dispersion_measure,
    print_color_examples,
    parse_time,
    info_string,
    akaike_information_criterion,
)


def test_taylor_horner_basic():
    """Check basic calculation against schoolbook formula."""
    assert taylor_horner(2.0, [10]) == 10
    assert taylor_horner(2.0, [10, 3]) == 10 + 3 * 2.0
    assert taylor_horner(2.0, [10, 3, 4]) == 10 + 3 * 2.0 + 4 * 2.0**2 / 2.0
    assert taylor_horner(
        2.0, [10, 3, 4, 12]
    ) == 10 + 3 * 2.0 + 4 * 2.0**2 / 2.0 + 12 * 2.0**3 / (3.0 * 2.0)


contents = """Random text file

with some stuff

"""


def test_open_or_use_string_write():
    """Check writing to filename."""
    with NamedTemporaryFile("w") as w:
        w.write(contents)
        w.flush()
        with open_or_use(w.name) as r:
            assert r.read() == contents


def test_open_or_use_string_read():
    """Check reading from filename."""
    with NamedTemporaryFile("r") as r:
        with open_or_use(r.name, "w") as w:
            w.write(contents)
        assert r.read() == contents


def test_open_or_use_file_write():
    """Check writing to file."""
    with NamedTemporaryFile("w") as wo:
        with open_or_use(wo) as w:
            w.write(contents)
        wo.flush()
        assert open(wo.name).read() == contents


def test_open_or_use_file_read():
    """Check reading from file."""
    with NamedTemporaryFile("r") as ro:
        with open(ro.name, "w") as w:
            w.write(contents)
        with open_or_use(ro) as r:
            assert r.read() == contents


@pytest.mark.parametrize("contents", ["", " ", "\n", "aa", "a\na", contents])
def test_lines_of(contents):
    """Check we get the same lines back through various means."""
    lines = contents.splitlines(True)
    assert list(lines_of(lines)) == lines
    with NamedTemporaryFile("w") as w:
        w.write(contents)
        w.flush()
        assert list(lines_of(w.name)) == lines
        with open(w.name) as r:
            assert list(lines_of(r)) == lines


@pytest.mark.parametrize(
    "lines, goodlines, comments",
    [
        ([" text stuff \n"], ["text stuff"], None),
        ([" text stuff \n\n"], ["text stuff"], None),
        ([" text stuff "], ["text stuff"], None),
        (["a\n", "\n", "\n", "b"], ["a", "b"], None),
        ([" text stuff \n"] * 7, ["text stuff"] * 7, None),
        (["\ttext stuff \n"], ["text stuff"], None),
        (["#\ttext stuff \n"], [], "#"),
        (["  #\ttext stuff \n"], [], "#"),
        (["C \ttext stuff \n"], [], "C "),
        (["  C \ttext stuff \n"], [], "C "),
        (["C\ttext stuff \n"], ["C\ttext stuff"], "C "),
        (["#\ttext stuff \n"], [], ("#", "C ")),
        (["C \ttext stuff \n"], [], ("#", "C ")),
        (["C \ttext stuff \n"], [], ["#", "C "]),
        (["C\ttext stuff \n"], ["C\ttext stuff"], ("#", "C ")),
        (["C\ttext stuff \n"], [], ("#", "C ", "C\t")),
    ],
)
def test_interesting_lines(lines, goodlines, comments):
    """Check various patterns of text and comments."""
    assert list(interesting_lines(lines, comments=comments)) == goodlines


def test_interesting_lines_input_validation():
    """Check it lets the user know about invalid comment markers."""
    with pytest.raises(ValueError):
        for _ in interesting_lines([""], comments=" C "):
            pass


def test_posvel_rejects_misshapen_quantity():
    with pytest.raises((ValueError, TypeError)):
        PosVel(1 * u.m, 1 * u.m / u.s)


def test_posvel_respects_label_constraints():
    p1 = [1, 0, 0]
    p2 = [0, 1, 0]
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]

    earth_mars = PosVel(p1, v1, origin="earth", obj="mars")
    mars_earth = PosVel(p2, v2, origin="mars", obj="earth")

    pv = earth_mars + mars_earth
    assert pv.origin == pv.obj == "earth"
    pv = mars_earth + earth_mars
    assert pv.origin == pv.obj == "mars"

    with pytest.raises(ValueError):
        pv = earth_mars - mars_earth

    with pytest.raises(ValueError):
        PosVel(p2, v2, origin="else")
    with pytest.raises(ValueError):
        PosVel(p2, v2, obj="else")


@composite
def posvel_arrays(draw):
    s = draw(array_shapes())
    dtype = draw(scalar_dtypes())
    pos = draw(arrays(dtype=dtype, shape=(3,) + s))
    vel = draw(arrays(dtype=dtype, shape=(3,) + s))
    return pos, vel


@composite
def posvel_arrays_and_indices(draw):
    pos, vel = draw(posvel_arrays())
    ix = tuple(draw(slices(n)) for n in pos.shape[1:])
    return pos, vel, ix


@given(posvel_arrays_and_indices())
def test_posvel_slice_indexing(pos_vel_ix):
    pos, vel, ix = pos_vel_ix
    pv = PosVel(pos, vel)
    pvs = pv[ix]
    ix_full = (slice(None, None, None),) + ix
    assert_array_equal(pvs.pos, pos[ix_full])
    assert_array_equal(pvs.vel, vel[ix_full])


def test_posvel_different_lengths_raises():
    pos = np.random.randn(3, 4, 5)
    vel = np.random.randn(3, 6)
    with pytest.raises(ValueError):
        PosVel(pos, vel)


@composite
def posvel_arrays_broadcastable(draw):
    s, s_pos, s_vel = draw(broadcastable_subshapes(array_shapes()))
    dtype = draw(scalar_dtypes())
    pos = draw(arrays(dtype=dtype, shape=(3,) + tuple(s_pos)))
    vel = draw(arrays(dtype=dtype, shape=(3,) + tuple(s_vel)))
    return pos, vel, (3,) + s


@composite
def boolean_subsets(draw, set_arrays):
    a = np.array(draw(set_arrays), dtype=bool)
    n = int(np.sum(a))
    r = np.zeros_like(a)
    if n > 0:
        r[a] = draw(arrays(just(bool), just((n,))))
    return a


@composite
def broadcastable_subshapes(draw, shapes):
    s = draw(shapes)
    b = draw(arrays(just(bool), just((len(s),))))
    b_pos = draw(boolean_subsets(just(b)))
    b_vel = b & (~b_pos)

    s_pos = np.array(s)
    s_pos[b_pos] = 1
    s_vel = np.array(s)
    s_vel[b_vel] = 1

    return s, tuple(int(s) for s in s_pos), tuple(int(s) for s in s_vel)


@given(posvel_arrays_broadcastable())
def test_posvel_broadcasts(pos_vel_shape):
    pos, vel, shape = pos_vel_shape
    pv = PosVel(pos, vel)
    assert pv.pos.shape == pv.vel.shape == shape


@given(
    posvel_arrays_broadcastable(),
    sampled_from([u.m, u.pc]),
    sampled_from([u.s, u.year]),
)
def test_posvel_broadcast_retains_quantity(pos_vel_shape, l_unit, t_unit):
    pos, vel, shape = pos_vel_shape
    pv = PosVel(pos * l_unit, vel * l_unit / t_unit)
    assert pv.pos.shape == pv.vel.shape == shape
    assert pv.pos.unit == l_unit
    assert pv.vel.unit == l_unit / t_unit


def test_posvel_reject_bogus_sizes():
    with pytest.raises(ValueError):
        PosVel([1, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        PosVel([1, 0, 0], [1, 0, 0, 0])
    with pytest.raises(ValueError):
        PosVel(np.array([1, 0]) * u.m, [1, 0, 0])
    with pytest.raises(ValueError):
        PosVel([1, 0, 0, 0], np.array([1, 0]) * u.m)


def test_posvel_str_sensible():
    assert "->" in str(PosVel([1, 0, 0], [0, 1, 0], "earth", "mars"))
    assert "earth" in str(PosVel([1, 0, 0], [0, 1, 0], "earth", "mars"))
    assert "mars" in str(PosVel([1, 0, 0], [0, 1, 0], "earth", "mars"))
    assert "->" not in str(PosVel([1, 0, 0], [0, 1, 0]))
    assert "17" in str(PosVel([17, 0, 0], [0, 1, 0]))
    assert str(PosVel([17, 0, 0], [0, 1, 0])).startswith("PosVel(")


# Test that the simplified functions all behave well when handed arrays as well
# as singletons


@composite
def array_pair(draw, dtype1, elements1, dtype2, elements2):
    s = draw(array_shapes())
    a = draw(arrays(dtype=dtype1, shape=s, elements=elements1))
    b = draw(arrays(dtype=dtype2, shape=s, elements=elements2))
    return s, a, b


@composite
def array_pair_broadcast(draw, dtype1, elements1, dtype2, elements2):
    s, s_a, s_b = draw(broadcastable_subshapes(array_shapes()))
    a = draw(arrays(dtype=dtype1, shape=s_a, elements=elements1))
    b = draw(arrays(dtype=dtype2, shape=s_b, elements=elements2))
    return s, a, b


@composite
def mjd_strs(draw):
    i = draw(integers(min_value=40000, max_value=60000))
    f = draw(floats(0, 1, allow_nan=False))
    return mjds_to_str(i, f)


@given(
    one_of(
        array_pair(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
        array_pair_broadcast(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
    )
)
def test_mjds_to_str_array(sif):
    s, i, f = sif
    r = mjds_to_str(i, f)
    assert hasattr(r, "dtype")
    assert np.shape(r) == s
    for r_i, i_i, f_i in np.nditer([r, i, f], flags=["refs_ok"]):
        assert r_i == mjds_to_str(i_i, f_i)


@given(
    one_of(
        array_pair(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
        array_pair_broadcast(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
    )
)
def test_mjds_to_str_array_roundtrip_doesnt_crash(sif):
    s, i, f = sif
    assume(s != ())
    str_to_mjds(mjds_to_str(i, f))


@given(
    one_of(
        array_pair(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
        array_pair_broadcast(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
    )
)
def test_mjds_to_str_array_roundtrip_close(sif):
    s, i, f = sif
    i_o, f_o = str_to_mjds(mjds_to_str(i, f))

    assert hasattr(i_o, "dtype")
    assert hasattr(f_o, "dtype")
    l = i.astype(np.longdouble) + f.astype(np.longdouble)
    l_o = i_o.astype(np.longdouble) + f_o.astype(np.longdouble)

    assert np.all(np.abs(l - l_o) * 86400 < 1e-9)


def test_mjds_to_str_singleton():
    assert isinstance(mjds_to_str(40000, 0.0), (str, bytes))


def test_str_to_mjds_singleton():
    jd1, jd2 = str_to_mjds("41498.0")
    assert isinstance(jd1, float)
    assert isinstance(jd2, float)


def test_str_to_mjds_exponential():
    assert str_to_mjds("4.1498e4") == str_to_mjds("41498")


def test_str_to_mjds_exponential_negative():
    str_to_mjds("4.1498e-4")


def test_str_to_mjds_exponential_fortran():
    assert str_to_mjds("4.1498d4") == str_to_mjds("41498")


def test_str_to_mjds_singleton_arrayobj():
    s = np.array(["41498.0"])[0]
    assert isinstance(s, str)
    jd1, jd2 = str_to_mjds(s)
    assert isinstance(jd1, float)
    assert isinstance(jd2, float)


def test_mjds_to_jds_singleton():
    jd1, jd2 = mjds_to_jds(40000, 0.0)
    assert isinstance(jd1, float)
    assert isinstance(jd2, float)


@given(arrays(object, array_shapes(), elements=mjd_strs()))
def test_str_to_mjds_array(s):
    i, f = str_to_mjds(s)
    assert np.shape(i) == np.shape(f) == np.shape(s)
    for i_i, f_i, s_i in np.nditer([i, f, s], flags=["refs_ok"]):
        assert i_i, f_i == str_to_mjds(s_i)


@given(
    one_of(
        array_pair(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
        array_pair_broadcast(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
    )
)
def test_mjds_to_jds_array(sif):
    s, i, f = sif
    jd1, jd2 = mjds_to_jds(i, f)
    assert np.shape(jd1) == np.shape(jd2) == s
    for jd1_i, jd2_i, i_i, f_i in np.nditer([jd1, jd2, i, f]):
        assert jd1_i, jd2_i == mjds_to_jds(i_i, f_i)


@given(
    one_of(
        array_pair(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
        array_pair_broadcast(
            np.int64,
            integers(min_value=40000, max_value=60000),
            float,
            floats(0, 1, allow_nan=False),
        ),
    )
)
def test_mjds_to_jds_pulsar_array(sif):
    s, i, f = sif
    jd1, jd2 = mjds_to_jds_pulsar(i, f)
    assert np.shape(jd1) == np.shape(jd2) == s
    for jd1_i, jd2_i, i_i, f_i in np.nditer([jd1, jd2, i, f]):
        assert jd1_i, jd2_i == mjds_to_jds_pulsar(i_i, f_i)


@given(
    one_of(
        array_pair(
            np.int64,
            integers(min_value=2440000, max_value=2460000),
            float,
            floats(0, 1, allow_nan=False),
        ),
        array_pair_broadcast(
            np.int64,
            integers(min_value=2440000, max_value=2460000),
            float,
            floats(0, 1, allow_nan=False),
        ),
    )
)
@example(s12=((1,), np.array([2440000]), np.array([0.0])))
def test_jds_to_mjds_array(s12):
    s, jd1, jd2 = s12
    i, f = jds_to_mjds(jd1, jd2)
    assert np.shape(f) == s
    assert np.shape(i) == s
    for jd1_i, jd2_i, i_i, f_i in np.nditer([jd1, jd2, i, f]):
        assert i_i, f_i == jds_to_mjds(jd1_i, jd2_i)


@given(
    one_of(
        array_pair(
            np.int64,
            integers(min_value=2440000, max_value=2460000),
            float,
            floats(0, 1, allow_nan=False),
        ),
        array_pair_broadcast(
            np.int64,
            integers(min_value=2440000, max_value=2460000),
            float,
            floats(0, 1, allow_nan=False),
        ),
    )
)
def test_jds_to_mjds_pulsar_array(s12):
    s, jd1, jd2 = s12
    i, f = jds_to_mjds_pulsar(jd1, jd2)
    assert np.shape(f) == s
    assert np.shape(i) == s
    for jd1_i, jd2_i, i_i, f_i in np.nditer([jd1, jd2, i, f]):
        assert i_i, f_i == jds_to_mjds_pulsar(jd1_i, jd2_i)


# pulsar_mjd and related formats


@pytest.mark.parametrize(
    "format_, type_",
    [
        ("mjd", float),
        ("pulsar_mjd", float),
        ("mjd_long", np.longdouble),
        ("pulsar_mjd_long", np.longdouble),
        ("mjd_string", (str, bytes)),
        ("pulsar_mjd_string", (str, bytes)),
    ],
)
def test_singleton_type(format_, type_):
    t = Time.now()
    assert isinstance(getattr(t, format_), type_)
    t.format = format_
    assert isinstance(t.value, type_)


@pytest.mark.parametrize(
    "format_, val, val2",
    [
        ("mjd", 40000, 1e-10),
        ("pulsar_mjd", 40000, 1e-10),
        ("mjd_long", np.longdouble(40000) + np.longdouble(1e-10), None),
        ("mjd_long", np.longdouble(40000), np.longdouble(1e-10)),
        ("pulsar_mjd_long", np.longdouble(40000) + np.longdouble(1e-10), None),
        ("pulsar_mjd_long", np.longdouble(40000), np.longdouble(1e-10)),
        ("mjd_string", "40000.0000000001", None),
        ("pulsar_mjd_string", "40000.0000000001", None),
    ],
)
def test_singleton_import(format_, val, val2):
    Time(val=val, val2=val2, format=format_, scale="utc")


# time_to


@pytest.mark.parametrize("format_", ["mjd", "pulsar_mjd"])
def test_time_to_longdouble_types(format_):
    t = Time.now()
    t.format = format_
    assert isinstance(time_to_longdouble(t), np.longdouble)

    t2 = Time(val=50000.0, val2=np.linspace(0, 1, 10), format=format_, scale="utc")
    assert time_to_longdouble(t2).dtype == np.longdouble


@pytest.mark.parametrize(
    "format_, val",
    product(
        ["mjd_string", "pulsar_mjd_string"],
        [1, False, lambda: False, {1: 2, 3: 4}, {1, 2, 3, 4}],
    ),
)
def test_mjd_string_bogus_types(format_, val):
    with pytest.raises(ValueError):
        Time(val=val, format=format_, scale="utc")


@pytest.mark.parametrize("format_", ["mjd", "pulsar_mjd"])
def test_mjd_string_rejects_val2(format_):
    with pytest.raises(ValueError):
        Time(val="58000", val2="foo", format=format_, scale="utc")


def test_time_from_mjd_string_rejects_other_formats():
    with pytest.raises(ValueError):
        time_from_mjd_string("58000", format="cxcsec")


def test_dmxparse():
    """Test for dmxparse function."""
    m = tm.get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    t = toa.get_TOAs(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim"))
    f = fitter.GLSFitter(toas=t, model=m)
    f.fit_toas()
    dmx = dmxparse(f, save=False)
    # make sure the start and end are not the same
    assert ((dmx["r2s"] - dmx["r1s"] > 0)).all()
    # Check exception handling
    m = tm.get_model(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_DMX.par"))
    t = toa.get_TOAs(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim"))
    f = fitter.WLSFitter(toas=t, model=m)
    f.fit_toas()
    dmx = dmxparse(f, save=False)


def test_dmxparse_write():
    # check output
    m = tm.get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    t = toa.get_TOAs(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim"))
    f = fitter.GLSFitter(toas=t, model=m)
    f.fit_toas()
    w = io.StringIO()
    dmx = dmxparse(f, save=w)
    w.seek(0)
    assert len(w.read()) > 0


def test_dmxparse_write_default():
    # check output to default filename
    m = tm.get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    t = toa.get_TOAs(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim"))
    f = fitter.GLSFitter(toas=t, model=m)
    f.fit_toas()
    dmx = dmxparse(f, save=True)
    with open("dmxparse.out") as r:
        assert len(r.read()) > 0
    os.remove("dmxparse.out")


def test_pmtot():
    """Test pmtot calculation"""
    from pint.utils import pmtot

    # This is ecliptic
    m = tm.get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    # Replace with units when we are at numpy 1.17+
    assert np.isclose(pmtot(m).value, 6.056830627)
    # This is euqatorial
    m2 = tm.get_model(os.path.join(datadir, "PSR_J0218+4232.par"))
    # Replace with units when we are at numpy 1.17+
    assert np.isclose(pmtot(m2).value, 6.323257250021364)
    m2.remove_component("AstrometryEquatorial")
    with pytest.raises(AttributeError):
        pmtot(m2)


def test_ftest():
    """Test for FTest. Numbers from example test."""
    chi2_1 = 5116.3297879409574835
    dof_1 = 4961
    chi2_2 = 5110.749818644068647
    dof_2 = 4960
    ft = FTest(chi2_1, dof_1, chi2_2, dof_2)
    # Test against scipy F-CDF, hardcoded test value
    assert np.isclose(0.020000171879625623, ft)


@pytest.mark.parametrize("dof_1,dof_2,seed", [(12, 9, 0), (101, 100, 0), (405, 400, 0)])
def test_Ftest_statistical(dof_1, dof_2, seed):
    """Verify that the F test reports about the right number of false positives.

    The F test reports the probability that the chi-squared would decrease by the
    observed amount even if the model is not actually a better fit. So this test
    generates some fake data where the model isn't any better a fit, and asks
    how often the F test probability is less than some threshold (say 0.01). This
    should occur in about threshold fraction of trials. We check this against a
    binomial distribution; by construction this test should fail for 2% of seeds,
    so just retry with a different seed if it fails.
    """
    random = np.random.default_rng(0)
    Fs = []
    for _ in range(10000):
        x = random.standard_normal(dof_1)
        Fs.append(FTest((x**2).sum(), dof_1, (x[:dof_2] ** 2).sum(), dof_2))
    threshold = 0.01
    assert (
        scipy.stats.binom(len(Fs), threshold).ppf(0.01)
        < sum(F < threshold for F in Fs)
        < scipy.stats.binom(len(Fs), threshold).ppf(0.99)
    )


def test_Ftest_chi2_increase():
    assert FTest(100, 100, 101, 99) == 1


def test_Ftest_dof_same():
    assert np.isnan(FTest(100, 100, 100, 100))


@pytest.mark.parametrize(
    "x, coeffs, order",
    [
        (1.2, [2], 1),
        (0.1, [2, 3], 1),
        (-1, [2, 3, 5], 1),
        (1.1, [2], 2),
        (1.3, [2, 3, 4, 5], 2),
        (1.3, [2, 3, 4, 5], 4),
        (1.3, [2, 3, 4, 5, 6], 4),
        (1.5, [2], 10),
        (1.7, [2, 3, 4], 0),
    ],
)
def test_taylor_horner_deriv(x, coeffs, order):
    def f(x):
        return taylor_horner(x, coeffs)

    df = Derivative(f, n=order)
    assert_allclose(df(x), taylor_horner_deriv(x, coeffs, order), atol=1e-11)


@pytest.mark.parametrize(
    "x, coeffs",
    [
        (1.2, [2]),
        (0.1, [2, 3]),
        (-1, [2, 3, 5]),
        (1.1, [2]),
        (1.3, [2, 3, 4, 5]),
        (1.5, [2]),
        (1.7, [2, 3, 4]),
    ],
)
def test_taylor_horner_equals_deriv(x, coeffs):
    assert_allclose(taylor_horner(x, coeffs), taylor_horner_deriv(x, coeffs, 0))


@pytest.mark.parametrize(
    "x, result, n",
    [(1 * u.s, 1 * u.m, 5), (1 * u.s, 1 * u.m, 1), (1 * u.km**2, 1 * u.m, 3)],
)
def test_taylor_horner_units_ok(x, result, n):
    coeffs = [result / x**i for i in range(n + 1)]
    taylor_horner(x, coeffs) + result


def test_list_parameters():
    list_parameters()


@given(
    tuples(binary(max_size=1_000_000), binary(max_size=1_000_000)).filter(
        lambda t: t[0] != t[1]
    )
)
def test_compute_hash_detects_changes(a_b):
    a, b = a_b
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        f = tmp_path / "file"
        f.write_bytes(a)
        h_a = compute_hash(f)
        f.write_bytes(b)
        h_b = compute_hash(f)
        assert h_a != h_b


@given(binary(max_size=1_000_000))
def test_compute_hash_accepts_no_change(a):
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        f = tmp_path / "file"
        f.write_bytes(a)
        h_a = compute_hash(f)

        g = tmp_path / "file2"
        g.write_bytes(a)
        h_b = compute_hash(g)

    assert h_a == h_b


def test_find_dmx():
    par = """
    PSR J1234+5678
    F0 1
    DM 10
    ELAT 10
    ELONG 0
    PEPOCH 54000
    DMXR1_0001 54000
    DMXR2_0001 55000
    DMX_0001 1
    DMXR1_0002 55000
    DMXR2_0002 56000
    DMX_0002 2
    """

    model = tm.get_model(io.StringIO(par))
    assert find_prefix_bytime(model, "DMX", 54500) == 1
    assert len(find_prefix_bytime(model, "DMX", 53500)) == 0


def test_merge_dmx():
    par = """
    PSR J1234+5678
    F0 1
    DM 10
    ELAT 10
    ELONG 0
    PEPOCH 54000
    DMXR1_0001 54000
    DMXR2_0001 55000
    DMX_0001 1
    DMXR1_0002 55000
    DMXR2_0002 56000
    DMX_0002 2
    """

    model = tm.get_model(io.StringIO(par))
    newindex = merge_dmx(model, 1, 2, value="mean")
    print(model, newindex)
    assert getattr(model, f"DMX_{newindex:04d}").value == 1.5


def test_convert_dm():
    dm = 10 * dmu
    dm_codata = convert_dispersion_measure(dm)

    assert np.isfinite(dm_codata)


def test_print_color_examples():
    print_color_examples()


@pytest.mark.parametrize(
    "t",
    [
        Time(55555, format="pulsar_mjd", scale="tdb", precision=9),
        55555 * u.d,
        55555.0,
        55555,
        "55555",
    ],
)
def test_parse_time(t):
    assert parse_time(t, scale="tdb") == Time(
        55555, format="pulsar_mjd", scale="tdb", precision=9
    )


def test_info_str():
    info = info_string()
    dinfo = info_string(detailed=True)


def test_aic():
    m = tm.get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    t = toa.get_TOAs(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim"))

    assert np.isfinite(akaike_information_criterion(m, t))
