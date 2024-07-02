from astropy import units as u, constants as c
from astropy.time import Time
import numpy as np
from pint.models import get_model_and_toas
import pytest
import os
from pinttestdata import datadir


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
