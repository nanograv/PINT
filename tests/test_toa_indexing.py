from io import StringIO

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import slices, integers, one_of, lists
from hypothesis.extra.numpy import arrays, array_shapes

from pint.toa import get_TOAs

tim = """FORMAT 1
fake 1400 54000 1.0 @ -flag thing
fake 1400 54001 1.0 @ -flag thing
fake 1400 54002 1.0 @ -flag other_thing
fake 1400 54003 2.0 @ -flag thing
fake 1400 54004 1.0 ao -flag thing
fake 1400 54005 1.0 @ -flag thing
fake 1400 54006 1.0 ao -flag thing
fake 1500 54007 1.0 ao -flag thing
fake 1500 54008 1.0 @ -flag thing -other_flag another_thing
"""
n_tim = len(tim.split("\n")) - 2


@pytest.mark.parametrize("high_precision", [True, False])
def test_get_TOAs(high_precision):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds(high_precision=high_precision)
    assert isinstance(m, np.ndarray)
    assert len(m) == n_tim
    assert np.all(np.diff(m) > 0)


@given(arrays(bool, n_tim))
def test_select(c):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds()
    assert len(toas) == len(c)
    toas.select(c)
    assert len(toas) == np.sum(c)
    assert np.all(toas.get_mjds() == m[c])
    if len(toas) > 0:
        toas.get_summary()


@given(arrays(bool, n_tim))
def test_getitem_boolean(c):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds()
    assert len(toas) == len(c)
    s = toas[c]
    assert len(s) == np.sum(c)
    assert np.all(s.get_mjds() == m[c])
    if len(s) > 0:
        toas.get_summary()


@given(
    one_of(
        arrays(int, array_shapes(max_dims=1), elements=integers(0, n_tim - 1)),
        lists(integers(0, n_tim - 1)),
        integers(0, n_tim - 1),
    )
)
def test_getitem_where(a):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds()
    s = toas[a]
    if not isinstance(a, int):
        assert len(s) == len(a)
        assert set(s.get_mjds()) == set(m[a])
    else:
        assert len(s) == 1
        assert set(s.get_mjds()) == set(m[[a]])
    if len(s) > 0:
        toas.get_summary()


@given(slices(n_tim))
def test_getitem_slice(c):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds()
    s = toas[c]
    assert set(s.get_mjds()) == set(m[c])
    assert (toas[c].get_mjds() == toas.get_mjds()[c]).all()
    if len(s) > 0:
        toas.get_summary()


def test_flag_column_reading():
    toas = get_TOAs(StringIO(tim), ephem="de421")
    assert (toas["flag"] == "other_thing").sum() == 1
    assert (toas["flag"] == "thing").sum() == len(toas) - 1
    assert (toas["other_flag"] == "another_thing").sum() == 1
    assert (toas["other_flag"] == "").sum() == len(toas) - 1


@pytest.mark.parametrize(
    "subset",
    [slice(1, n_tim), np.array([False] + [True] * (n_tim - 1)), np.arange(1, n_tim)],
)
def test_flag_column_reading_subset(subset):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    assert (toas["flag", subset] == "other_thing").sum() == 1
    assert (toas["flag", subset] == "thing").sum() == len(toas[subset]) - 1
    assert (toas["other_flag", subset] == "another_thing").sum() == 1
    assert (toas["other_flag", subset] == "").sum() == len(toas[subset]) - 1


def test_flag_column_writing():
    toas = get_TOAs(StringIO(tim), ephem="de421")
    toas["new_flag"] = "new_value"
    assert np.all(toas["new_flag"] == "new_value")
    toas["new_flag", 0] = ""
    assert (toas["new_flag"] == "new_value").sum() == len(toas) - 1
    toas["new_flag"] = ""
    assert np.all(toas["new_flag"] == "")
