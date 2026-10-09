import io

import pytest

from pint.models import get_model

input_par = """PSRJ                           J0523-7125
EPHEM                               DE405
CLK                               TT(TAI)
UNITS                                 TDB
START              55415.8045121523831364
FINISH             59695.2673406681377430
TIMEEPH                              FB90
T2CMETHOD                        IAU2000B
DILATEFREQ                              N
DMDATA                                  N
NTOA                                   87
CHI2                   404.35757416343705
RAJ                      5:23:48.66000000
DECJ                   -71:25:52.58000000
PMRA                                  0.0
PMDEC                                 0.0
PX                                    0.0
POSEPOCH           59369.0000000000000000
F0                  3.1001291305547288772 1 2.7544353718657238425e-11
F1              -2.4892219423278130317e-15 1 2.8277388169449218064e-19
PEPOCH             59609.0000000000000000
"""


def test_prefixparaminheritance_stayfrozen():
    # start with the case that has frozen parameters.  make sure they remain so
    m = get_model(io.StringIO(input_par + "\nF2 0\nF3 0"))
    assert m.F2.frozen
    assert m.F3.frozen


def test_prefixparaminheritance_unfrozen():
    # start with the case that has the new parameters unfrozen
    m = get_model(io.StringIO(input_par + "\nF2 0 1\nF3 0 1"))
    assert not m.F2.frozen
    assert not m.F3.frozen


# Template parameters (the first of a family) that PINT creates unfrozen,
# with the other par lines they need
_templates = {
    "DMX_0001": "DMX 14\nDMXR1_0001 59000\nDMXR2_0001 59010\n",
    "WXSIN_0001": "WXEPOCH 59000\nWXFREQ_0001 0.01\nWXCOS_0001 0\n",
    "CMX_0001": "CMEPOCH 59000\nTNCHROMIDX 4\nCMXR1_0001 59000\nCMXR2_0001 59010\n",
    "JUMP -fe L": "",
}


@pytest.mark.parametrize("name", list(_templates))
@pytest.mark.parametrize(
    "rest, frozen",
    [
        ("1e-6", True),
        ("1e-6 2e-7", True),
        ("1e-6 0 2e-7", True),
        ("1e-6 1 2e-7", False),
    ],
)
def test_template_param_frozen_without_fit_flag(name, rest, frozen):
    # A parameter without a fit flag is frozen
    m = get_model(io.StringIO(input_par + _templates[name] + f"{name} {rest}\n"))
    param = m.JUMP1 if name.startswith("JUMP") else getattr(m, name)
    assert param.value == pytest.approx(1e-6)
    assert param.frozen is frozen


@pytest.mark.parametrize("rest, unc", [("2e-6", 3e-7), ("2e-6 4e-7", 4e-7)])
def test_mask_param_frozen_without_fit_flag(rest, unc):
    # reading a line without a fit flag into a free mask parameter freezes it
    m = get_model(io.StringIO(input_par + "JUMP -fe L 1e-6 1 3e-7\n"))
    assert not m.JUMP1.frozen
    assert m.JUMP1.from_parfile_line(f"JUMP -fe L {rest}")
    assert m.JUMP1.frozen
    assert m.JUMP1.value == pytest.approx(2e-6)
    assert m.JUMP1.uncertainty_value == pytest.approx(unc)
