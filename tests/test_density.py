import astropy.units as u
import pint.toa as toa
import pint.simulation as simulation
from pint.models import get_model
import io
import pytest


@pytest.mark.parametrize("ndays", [7 * u.d, 20 * u.d])
def test_get_highest_density_range(ndays):
    par_base = """
    PSR J1234+5678
    F0 1 0
    ELAT 0 0
    ELONG 0 0
    PEPOCH 57000
    DM 10 0
    SOLARN0 0
    """
    model = get_model(io.StringIO(par_base))
    toas_1 = simulation.make_fake_toas_uniform(57000, 58000, 1000, model, obs="@")
    toas_2 = simulation.make_fake_toas_uniform(
        57500, 57500 + ndays.value, 100, model, obs="@"
    )
    merged = toa.merge_TOAs([toas_1, toas_2])
    if ndays == 7 * u.d:
        x1 = merged.get_highest_density_range()
    x2 = merged.get_highest_density_range(ndays)

    assert abs(x2[0].value - 57500) <= 1e-5
    assert abs(x2[1].value - (57500 + ndays.value)) <= 1e-5
    if ndays == 7 * u.d:
        assert abs(x2[0].value - x1[0].value) <= 1e-5
        assert abs(x2[1].value - x1[1].value) <= 1e-5
