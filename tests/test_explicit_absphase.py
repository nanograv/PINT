from pint.models import get_model, get_model_and_toas
from pint.simulation import make_fake_toas_uniform
from pint.residuals import Residuals
from io import StringIO
import pytest

par = """
F0      100
PEPOCH  50000
"""


@pytest.fixture
def fake_toas():
    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(50000, 50100, 50, m, add_noise=True)
    return t


@pytest.fixture
def model():
    return get_model(StringIO(par))


def test_add_tzr_toa(model, fake_toas):
    assert "AbsPhase" not in model.components

    model.add_tzr_toa(fake_toas)

    assert "AbsPhase" in model.components
    assert hasattr(model, "TZRMJD")
    assert hasattr(model, "TZRSITE")
    assert hasattr(model, "TZRFRQ")

    with pytest.raises(ValueError):
        model.add_tzr_toa(fake_toas)


@pytest.mark.parametrize("use_abs_phase", [True, False])
def test_residuals(model, fake_toas, use_abs_phase):
    res = Residuals(fake_toas, model, use_abs_phase=use_abs_phase)
    res.calc_phase_resids()
    assert ("AbsPhase" in model.components) == use_abs_phase


@pytest.mark.parametrize("use_abs_phase", [True, False])
def test_residuals_2(model, fake_toas, use_abs_phase):
    res = Residuals(fake_toas, model, use_abs_phase=False)
    res.calc_phase_resids(use_abs_phase=use_abs_phase)
    assert ("AbsPhase" in model.components) == use_abs_phase


@pytest.mark.parametrize("add_tzr", [True, False])
def test_get_model(fake_toas, add_tzr):
    toas_for_tzr = fake_toas if add_tzr else None
    m = get_model(StringIO(par), toas_for_tzr=toas_for_tzr)
    assert ("AbsPhase" in m.components) == add_tzr


@pytest.mark.parametrize("add_tzr", [True, False])
def test_get_model_and_toas(fake_toas, add_tzr):
    timfile = "fake_toas.tim"
    fake_toas.write_TOA_file(timfile)

    m, t = get_model_and_toas(StringIO(par), timfile, add_tzr_to_model=add_tzr)

    assert ("AbsPhase" in m.components) == add_tzr
