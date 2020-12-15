from io import StringIO

import numpy as np
import pytest
from hypothesis import given, assume
from hypothesis.strategies import slices, integers, booleans, one_of, lists
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
fake 1500 54008 1.0 @ -flag thing
"""
n_tim = len(tim.split("\n")) - 2


@pytest.mark.parametrize("high_precision", [True, False])
def test_get_TOAs(high_precision):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds(high_precision=high_precision)
    assert isinstance(m, np.ndarray)
    assert not np.all(
        np.diff(m) > 0
    ), "returned values should be grouped by observatory"
    assert len(m) == n_tim


@given(arrays(bool, n_tim))
def test_select(c):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds()
    assert len(toas) == len(c)
    toas.select(c)
    assert len(toas) == np.sum(c)
    assert np.all(toas.get_mjds() == m[c])
    if len(toas) > 0:
        assert np.all(
            toas.table["mjd_float"] == toas.table.group_by("obs")["mjd_float"]
        )
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
        assert np.all(s.table["mjd_float"] == s.table.group_by("obs")["mjd_float"])
        toas.get_summary()


@given(
    one_of(
        arrays(int, array_shapes(max_dims=1), elements=integers(0, n_tim - 1)),
        lists(integers(0, n_tim - 1)),
    )
)
def test_getitem_where(a):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds()
    s = toas[a]
    assert len(s) == len(a)
    assert set(s.get_mjds()) == set(m[a])
    if len(s) > 0:
        assert np.all(s.table["mjd_float"] == s.table.group_by("obs")["mjd_float"])
        toas.get_summary()


@given(slices(n_tim))
def test_getitem_slice(c):
    toas = get_TOAs(StringIO(tim), ephem="de421")
    m = toas.get_mjds()
    s = toas[c]
    assert set(s.get_mjds()) == set(m[c])
    if len(s) > 0:
        assert np.all(s.table["mjd_float"] == s.table.group_by("obs")["mjd_float"])
        toas.get_summary()
