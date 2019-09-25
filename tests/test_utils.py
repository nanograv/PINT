"""Test basic functionality of the :module:`pint.utils`."""

from __future__ import absolute_import, division, print_function

from tempfile import NamedTemporaryFile

from pint.utils import open_or_use, taylor_horner, lines_of, interesting_lines
import astropy.units as u
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import array_shapes, arrays, scalar_dtypes
from hypothesis.strategies import (
    composite,
    frozensets,
    integers,
    just,
    sampled_from,
    slices,
)
from numpy.testing import assert_array_equal

from pint.utils import PosVel, taylor_horner


def test_taylor_horner_basic():
    """Check basic calculation against schoolbook formula."""
    assert taylor_horner(2.0, [10]) == 10
    assert taylor_horner(2.0, [10, 3]) == 10 + 3 * 2.0
    assert taylor_horner(2.0, [10, 3, 4]) == 10 + 3 * 2.0 + 4 * 2.0 ** 2 / 2.0
    assert taylor_horner(
        2.0, [10, 3, 4, 12]
    ) == 10 + 3 * 2.0 + 4 * 2.0 ** 2 / 2.0 + 12 * 2.0 ** 3 / (3.0 * 2.0)


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
        ([" text stuff \n"] * 7, ["text stuff"] * 7, None),
        (["\ttext stuff \n"], ["text stuff"], None),
        (["#\ttext stuff \n"], [], "#"),
        (["  #\ttext stuff \n"], [], "#"),
        (["C \ttext stuff \n"], [], "C "),
        (["  C \ttext stuff \n"], [], "C "),
        (["C\ttext stuff \n"], ["C\ttext stuff"], "C "),
        (["#\ttext stuff \n"], [], ("#", "C ")),
        (["C \ttext stuff \n"], [], ("#", "C ")),
        (["C\ttext stuff \n"], ["C\ttext stuff"], ("#", "C ")),
    ],
)
def test_interesting_lines(lines, goodlines, comments):
    """Check various patterns of text and comments."""
    assert list(interesting_lines(lines, comments=comments)) == goodlines


def test_interesting_lines_input_validation():
    """Check it lets the user know about invalid comment markers."""
    with pytest.raises(ValueError):
        for l in interesting_lines([""], comments=" C "):
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
    pos = draw(arrays(dtype, (3,) + s))
    vel = draw(arrays(dtype, (3,) + s))
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
    pos = draw(arrays(dtype, (3,) + tuple(s_pos)))
    vel = draw(arrays(dtype, (3,) + tuple(s_vel)))
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
        PosVel([1,0],[1,0,0])
    with pytest.raises(ValueError):
        PosVel([1,0,0],[1,0,0,0])
    with pytest.raises(ValueError):
        PosVel(np.array([1,0])*u.m,[1,0,0])
    with pytest.raises(ValueError):
        PosVel([1,0,0,0], np.array([1,0])*u.m)

def test_posvel_str_sensible():
    assert "->" in str(PosVel([1,0,0],[0,1,0],"earth","mars"))
    assert "earth" in str(PosVel([1,0,0],[0,1,0],"earth","mars"))
    assert "mars" in str(PosVel([1,0,0],[0,1,0],"earth","mars"))
    assert "->" not in str(PosVel([1,0,0],[0,1,0]))
    assert "17" in str(PosVel([17,0,0],[0,1,0]))
    assert str(PosVel([17,0,0],[0,1,0])).startswith("PosVel(")
