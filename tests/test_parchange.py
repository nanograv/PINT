from astropy import units as u, constants as c
from astropy.time import Time
import numpy as np
from pint.models import get_model, get_model_and_toas
import pytest
import os
from io import StringIO
from pinttestdata import datadir


par1 = """
RAJ      17:48:04.53480592  0          0.00005548
DECJ     -24:46:34.6586741  0           0.0279874
F0              118.538254  1  0.0000000000054826
F1            8.485977D-15  0  9.543894501911D-20
PEPOCH        53000.000000
DM              237.062679  0            0.005850
SOLARN0              10.00
EPHEM               DE421
CLK                 UNCORR                          
UNITS               TDB
"""

par2 = """
RAJ      17:48:04.53480592  0          0.00005548
DECJ     -24:46:34.6586741  0           0.0279874
F0              666.666666  1  0.0000000000054826
F1            8.485977D-15  0  9.543894501911D-20
PEPOCH        53000.000000
DM              237.062679  0            0.005850
SOLARN0              10.00
EPHEM               DE421
CLK                 UNCORR                          
UNITS               TDB
"""


@pytest.mark.parametrize(
    "par,newvalue",
    (
        ("PLANET_SHAPIRO", True),
        ("F1", 10 * u.Hz / u.s),
        ("F0", 20),
        ("PSR", "ABC-XYZ"),
        ("POSEPOCH", 53760.0),
        ("PEPOCH", Time(53760, format="mjd")),
    ),
)
def test_parchange(par, newvalue):
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )
    setattr(m, par, newvalue)
    if isinstance(newvalue, (u.Quantity, Time)):
        assert m[par].quantity == newvalue
    else:
        assert m[par].value == newvalue


@pytest.mark.parametrize(
    "par,newvalue",
    (
        ("F1", 10 * u.Hz / u.s**2),
        ("F0", "abc"),
    ),
)
def test_parchange_fails(par, newvalue):
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )

    with pytest.raises((ValueError, u.UnitConversionError)):
        setattr(m, par, newvalue)


def test_parcopy():
    m1 = get_model(StringIO(par1))
    m2 = get_model(StringIO(par2))
    m1.F0 = m2.F0
    assert m1.F0 == m2.F0
    assert m1.components["Spindown"].F0 == m2.components["Spindown"].F0
    assert m1.as_parfile(include_info=False) == m2.as_parfile(include_info=False)


def test_parcopy_andchange():
    m1 = get_model(StringIO(par1))
    m2 = get_model(StringIO(par2))
    m1.F0 = m2.F0
    m2.F0.value = 20
    assert m1.F0 == m2.F0
    assert m1.components["Spindown"].F0 == m2.components["Spindown"].F0
    assert m1.as_parfile(include_info=False) == m2.as_parfile(include_info=False)


def test_parcopy_toplevel():
    m1 = get_model(StringIO(par1))
    m2 = get_model(StringIO(par2))
    m1.PSR.value = "ABC"
    m2.PSR.value = "DEF"
    m1.PSR = m2.PSR
    assert m1.PSR == m2.PSR
    m1.F0 = m2.F0
    assert m1.as_parfile(include_info=False) == m2.as_parfile(include_info=False)


def test_parcopy_toplevel_andchange():
    m1 = get_model(StringIO(par1))
    m2 = get_model(StringIO(par2))
    m1.PSR.value = "ABC"
    m2.PSR.value = "DEF"
    m1.PSR = m2.PSR
    assert m1.PSR == m2.PSR
    m2.PSR.value = "HGI"
    assert m1.PSR == m2.PSR
    m1.F0 = m2.F0
    assert m1.as_parfile(include_info=False) == m2.as_parfile(include_info=False)
