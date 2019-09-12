import pytest

from pint.models.timing_model import TimingModel
from pint.models.astrometry import AstrometryEquatorial
from pint.models.dispersion_model import DispersionDM, DispersionDMX


def test_forgot_name():
    with pytest.raises(ValueError):
        TimingModel(AstrometryEquatorial())
    with pytest.raises(ValueError):
        TimingModel([AstrometryEquatorial(), DispersionDM()])


@pytest.fixture
def model():
    return TimingModel(components=[AstrometryEquatorial(), DispersionDM(), DispersionDMX()])


def test_category_dict(model):
    d = model.components
    assert len(d) == 3
    #assert set(d.keys()) == set(T.component_types)
    #assert d==T.get_component_of_category()

def test_component_categories(model):
    for k, v in model.components.items():
        assert model.get_component_type(v) != v.category
