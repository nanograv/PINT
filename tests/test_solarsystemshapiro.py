from pint.models import get_model_and_toas
from pinttestdata import datadir
import pytest


def test_solarsystemshapiro_exception():
    m, t = get_model_and_toas(datadir / "NGC6440E.par", datadir / "NGC6440E.tim")
    m.PLANET_SHAPIRO.value = True

    with pytest.raises(KeyError):
        m.components["SolarSystemShapiro"].solar_system_shapiro_delay(t)
