"""Test model building and structure for simple models."""

from os.path import join
from tempfile import NamedTemporaryFile

import pytest

from pint.models.astrometry import AstrometryEquatorial
from pint.models.dispersion_model import DispersionDM, DispersionDMX
from pint.models.model_builder import UnknownBinaryModel, get_model, get_model_new
from pint.models.timing_model import MissingParameter, TimingModel
from pinttestdata import datadir


def test_forgot_name():
    """Check argument validation in case 'name' is forgotten."""
    with pytest.raises(ValueError):
        TimingModel(AstrometryEquatorial())
    with pytest.raises(ValueError):
        TimingModel([AstrometryEquatorial(), DispersionDM()])


@pytest.fixture
def model():
    """Make a simple model."""
    return TimingModel(
        components=[AstrometryEquatorial(), DispersionDM(), DispersionDMX()]
    )


def test_category_dict(model):
    """Check the different ways of grouping components."""
    d = model.components
    assert len(d) == 3
    # assert set(d.keys()) == set(T.component_types)
    # assert d==T.get_component_of_category()


def test_component_categories(model):
    """Check that the different groupings are different.

    Actually I expected them to be the same.

    """
    for k, v in model.components.items():
        assert model.get_component_type(v) != v.category


parfile = join(datadir, "J1744-1134.basic.par")
par_template = parfile + "\n" + "BINARY {}\n"


binary_models = [
    (get_model, "BT", pytest.raises(MissingParameter)),
    (get_model, "ELL1", pytest.raises(MissingParameter)),
    (get_model, "ELL1H", pytest.raises(MissingParameter)),
    (get_model_new, "BT", pytest.raises(MissingParameter)),
    (get_model_new, "ELL1", pytest.raises(MissingParameter)),
    (get_model_new, "ELL1H", pytest.raises(MissingParameter)),
    (get_model, "T2", pytest.raises(UnknownBinaryModel)),
    (get_model, "ELLL1", pytest.raises(UnknownBinaryModel)),
    (get_model_new, "T2", pytest.raises(UnknownBinaryModel)),
    (get_model_new, "ELLL1", pytest.raises(UnknownBinaryModel)),
]


@pytest.mark.parametrize("func, name, expectation", binary_models)
def test_valid_model(func, name, expectation):
    """Check handling of bogus binary models.

    Note that ``get_model_new`` currently reports different errors
    from the old ``get_model``.

    """
    with NamedTemporaryFile("w") as f:
        f.write(par_template.format(name))
        f.flush()
        with expectation:
            func(f.name)
