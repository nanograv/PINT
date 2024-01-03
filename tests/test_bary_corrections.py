from pint.models import get_model
from pint.toa import get_TOAs
from io import StringIO
import pytest

import astropy.units as u
import numpy as np

par = """
CLK        TT(BIPM2021)
EPHEM             DE421
RAJ             0:00:00
DECJ            0:00:00
UNITS               TDB
F0                  100
PEPOCH            60000
TZRMJD            60000
TZRSITE               @
TZRFRQ              inf
"""

tim = """
FORMAT 1
ref 0.0 60000.0 0.0 bat
coe 0.0 60000.1 0.0 coe
foo 0.0 60000.2 0.0 pks
bar 0.0 60000.3 0.0 lst
baz 0.0 60000.3 0.0 jb
"""

bary_tempo2 = [
    "60000.000000000000000",
    "60000.095531198261067",
    "60000.195527189099078",
    "60000.295522704703600",
    "60000.295522772151454",
]

phase_tempo2 = [
    "0.0",
    "0.552975622109443065710",
    "0.913816033558759954758",
    "0.168639107672788668424",
    "0.751388562124702730216",
]

bary_tempo2 = u.d * np.loadtxt(bary_tempo2, dtype=np.longdouble)
phase_tempo2 = (
    u.dimensionless_unscaled
    * np.loadtxt(phase_tempo2, dtype=np.longdouble)
    % (np.longdouble("1"))
)


@pytest.fixture
def model():
    return get_model(StringIO(par))


@pytest.fixture
def toas():
    return get_TOAs(StringIO(tim))


def test_first_toa_is_bat(toas):
    """Test if the ordering was respected, a condition for following tests."""
    assert (
        toas.__getitem__(index=0).to_TOA_list()[0].obs == "barycenter"
    ), "First TOA in the list should be barycentered."


def test_bat_toa_corrections(toas):
    """Test if a TOA defined in the barycenter gets the clock corrections."""
    failstr = "Clock corrections applied to TOA even if in: {0}.".format(
        toas.__getitem__(index=0).to_TOA_list()[0].obs
    )
    assert "clkcorr" not in toas.__getitem__(index=0).table["flags"][0], failstr


def test_bat_tzr_corrections(toas, model):
    """Test if a TZR defined in the barycenter gets the clock corrections."""

    # Make sure the component is untouched.
    assert model.components["AbsPhase"].tz_cache is None, "tz_cache already touched."

    tz = model.components["AbsPhase"].get_TZR_toa(toas)
    # At this point its tzr object should be True.
    assert tz.tzr, f"The TZR has tzr attribute: {tzr}."

    # Test wether clock corrections have been applied.
    failstr = "Clock corrections applied to TZR even if in: {0}.".format(
        tz.__getitem__(index=0).to_TOA_list()[0].obs
    )
    assert "clkcorr" not in tz.__getitem__(index=0).table["flags"][0], failstr


def test_tempo2_consistency(toas, model):
    """Test if derived BATs and absolute phases are consisent with those of tempo2."""
    bary_pint = model.get_barycentric_toas(toas)
    phase_pint = model.phase(toas, abs_phase=True)[1]
    phase_pint = phase_pint % 1

    # Tolerance of 1us, it is smaller than the typical 27us
    # from TT(BIPM).
    atol = (1.0 * u.us).to(u.d)

    bary_test = u.isclose(bary_tempo2, bary_pint, rtol=0, atol=atol)

    # The same for phases, times the frequency in the sample
    # par file.
    atol = (1.0 * u.us) * (100 * u.Hz)

    phase_test = u.isclose(phase_tempo2, phase_pint, rtol=0, atol=atol)

    delta_bary = bary_tempo2 - bary_pint
    delta_phase = phase_tempo2 - phase_pint

    for db, dp in zip(delta_bary, delta_phase):
        dbnano = db.to(u.ns)
        print(f"BAT delta: {db:+.6e}, {dbnano:+.6e}, phase {dp:+.6e}")

    for db, dp, BARY_TEST, PHASE_TEST in zip(
        delta_bary, delta_phase, bary_test, phase_test
    ):
        assert BARY_TEST, f"BAT delta is beyond 1us: {db:+.6e}"
        assert PHASE_TEST, f"PHASE delta is beyons 1us: {dp:+.6e}"
