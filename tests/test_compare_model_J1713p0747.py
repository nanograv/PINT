import pint.models.model_builder as mb
import pytest


@pytest.fixture()
def model_J1713p0747_ECL():
    return mb.get_model("datafile/J1713+0747_NANOGrav_11yv0_short.gls.par")


@pytest.fixture()
def model_J1713p0747_ICRS():
    return mb.get_model("datafile/J1713+0747_NANOGrav_11yv0_short.gls.ICRS.par")


def test_model_compare(model_J1713p0747_ECL, model_J1713p0747_ICRS):
    comparison1 = model_J1713p0747_ECL.compare(model_J1713p0747_ICRS)
    comparison2 = model_J1713p0747_ICRS.compare(model_J1713p0747_ECL)
    assert isinstance(comparison1, str) and isinstance(comparison2, str)
