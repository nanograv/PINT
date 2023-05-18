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
        "PLANET_SHAPIRO",
        "M2",
        "EDOT",
        "ECC",
        "OM",
        "T0",
        "TASC",
        "XDOT",
    ],
)
def test_par_units(p):
    unit = get_unit(p)
    print(f"{p}: {unit}")
    assert isinstance(unit, u.UnitBase) or unit is None


def test_par_units_fails():
    with pytest.raises(UnknownParameter):
        unit = get_unit("notapar")
