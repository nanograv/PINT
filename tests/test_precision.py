import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers, one_of

from pint.utils import longdouble2str, str2longdouble


@given(
    one_of(integers(40000, 60000), integers(-3000000, 3000000)),
    floats(-2, 2, allow_nan=False),
)
def test_longdouble_str_roundtrip_is_exact(i, f):
    ld = np.longdouble(i) + np.longdouble(f)
    assert ld == str2longdouble(longdouble2str(ld))

@given(
    one_of(integers(40000, 60000), integers(-3000000, 3000000)),
    floats(-2, 2, allow_nan=False),
)
def test_str_longdouble_roundtrip_is_exact(i, f):
    ld = np.longdouble(i) + np.longdouble(f)
    s = longdouble2str(ld)
    assert longdouble2str(str2longdouble(s)) == s

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


@pytest.mark.parametrize(
    "s", [b"", b"1", b"1.34", b"abd", 1.0, 1, None, {}, [], re.compile("fo+")]
)
def test_longdouble2str_rejects_non_strings(s):
    with pytest.raises(TypeError):
        str2longdouble(s)
