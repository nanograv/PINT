import pytest
from astropy import units as u
from pint.models.timing_model import UnknownParameter
from pint.utils import get_unit


@pytest.mark.parametrize(
    "p",
    [
        "F0",
        "F1",
        "DM",
        "DMX_0001",
        "DMX_0002",
        "DMXR1_0001",
        "POSEPOCH",
        "PMRA",
        "PMELONG",
        "FB0",
        "FB12",
        "PB",
        "RA",
        "A1",
        "M2",
        "EDOT",
        "ECC",
        "OM",
        "T0",
        "TASC",
        "XDOT",
        "EFAC",
        "EQUAD",
        "JUMP1",
    ],
)
def test_par_units(p):
    unit = get_unit(p)
    print(f"{p}: {unit}")
    assert isinstance(unit, u.UnitBase)


# strings should have units of None
@pytest.mark.parametrize(
    "p",
    [
        "PLANET_SHAPIRO",
    ],
)
def test_par_units_none(p):
    unit = get_unit(p)
    print(f"{p}: {unit}")
    assert unit is None


def test_par_units_fails():
    with pytest.raises(UnknownParameter):
        unit = get_unit("notapar")
