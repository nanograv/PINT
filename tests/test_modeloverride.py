import contextlib
import io
import os
import time
import pytest

import astropy.units as u
from astropy.time import Time
import numpy as np
from pint.models import get_model, get_model_and_toas
import pint.simulation


par = """PSRJ      J1636-5133
RAJ       16:35:44.7781433          1 0.05999748816321897513
DECJ      -51:34:18.01262           1 0.73332573676867170105
F0       2.9404155099936412855      1 0.00000000013195919743
F1       -1.4209854506981192501e-14 1 8.2230767370490607034e-17
PEPOCH 60000
DM       313
BINARY     ELL1
PB       0.74181505310937273631     1 0.00000018999923507341
A1       1.5231012457846993008      1 0.00050791514972366278
TASC      59683.784709068155703     1 0.00004690256150561100"""


# add in parameters that exist, parameters that don't, float, bool, and string.  Then somem quantities
@pytest.mark.parametrize(
    ("k", "v"),
    [
        ("F1", -2e-14),
        ("F2", 1e-12),
        ("PSR", "ABC"),
        ("DMDATA", True),
        ("F1", -1e-10 * u.Hz / u.day),
        ("F2", -1e-10 * u.Hz / u.day**2),
        ("PEPOCH", Time(55000, format="pulsar_mjd", scale="tdb")),
        ("JUMP", "mjd 55000 56000 0.03"),
    ],
)
def test_paroverride(k, v):
    kwargs = {k: v}
    m = get_model(io.StringIO(par), **kwargs)
    if isinstance(v, (str, bool)):
        if k != "JUMP":
            assert getattr(m, k).value == v
        else:
            assert getattr(m, f"{k}1").value == 0.03
            assert getattr(m, f"{k}1").key == "mjd"
            assert getattr(m, f"{k}1").key_value == [55000.0, 56000.0]
    elif isinstance(v, u.Quantity):
        assert np.isclose(getattr(m, k).quantity, v)
    elif isinstance(v, Time):
        assert getattr(m, k).quantity == v
    else:
        assert np.isclose(getattr(m, k).value, v)


# these should fail:
# adding F3 without F2
# adding an unknown parameter
# adding an improper value
# adding a value with incorrect units
@pytest.mark.parametrize(
    ("k", "v"), [("F3", -2e-14), ("TEST", -1), ("F1", "test"), ("F1", -1e-10 * u.Hz)]
)
def test_paroverride_fails(k, v):
    kwargs = {k: v}
    with pytest.raises((AttributeError, ValueError)):
        m = get_model(io.StringIO(par), **kwargs)


# add in parameters that exist, parameters that don't, float, and string
@pytest.mark.parametrize(("k", "v"), [("F1", -2e-14), ("F2", 1e-12), ("PSR", "ABC")])
def test_paroverride_withtim(k, v):
    kwargs = {k: v}
    m = get_model(io.StringIO(par), **kwargs)
    t = pint.simulation.make_fake_toas_uniform(50000, 58000, 20, model=m)
    o = io.StringIO()
    t.write_TOA_file(o)
    o.seek(0)
    m2, t2 = get_model_and_toas(io.StringIO(par), o, **kwargs)
    if isinstance(v, str):
        assert getattr(m2, k).value == v
    else:
        assert np.isclose(getattr(m2, k).value, v)
