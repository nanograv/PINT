"""Test model building and structure for simple models."""

from glob import glob
from os.path import basename, join
import pytest

import astropy.units as u
from pint.models.astrometry import AstrometryEquatorial
from pint.models.dispersion_model import DispersionDM, DispersionDMX
from pint.models.spindown import Spindown
from pint.models.model_builder import get_model
from pint.models.timing_model import (
    MissingParameter,
    TimingModel,
    Component,
    UnknownBinaryModel,
)
from pinttestdata import datadir


@pytest.fixture
def tmp_dir(tmpdir):
    yield str(tmpdir)


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
    # assert d==T.get_components_by_category()


def test_component_categories(model):
    """Check that the different groupings are different.

    Actually I expected them to be the same.

    """
    for k, v in model.components.items():
        assert model.get_component_type(v) != v.category


parfile = join(datadir, "J1744-1134.basic.par")
par_template = parfile + "\n" + "BINARY {}\n"


binary_models = [
    ("BT", pytest.raises(MissingParameter)),
    ("ELL1", pytest.raises(MissingParameter)),
    ("ELL1H", pytest.raises(MissingParameter)),
    ("T2", pytest.raises(UnknownBinaryModel)),
    ("ELLL1", pytest.raises(UnknownBinaryModel)),
    ("BTX", pytest.raises(UnknownBinaryModel)),
    ("DDFWHE", pytest.raises(UnknownBinaryModel)),
    ("MSS", pytest.raises(UnknownBinaryModel)),
]


@pytest.mark.parametrize("name, expectation", binary_models)
def test_valid_model(tmp_dir, name, expectation):
    """Check handling of bogus binary models.

    Note that ``get_model_new`` currently reports different errors
    from the old ``get_model``.

    """
    fn = join(tmp_dir, "file.par")
    with open(fn, "w") as f:
        f.write(par_template.format(name))
    with expectation:
        get_model(f.name)


@pytest.mark.xfail(
    reason="This parfile includes both ecliptic and equatorial coordinates"
)
def test_ecliptic():
    parfile = join(datadir, "J1744-1134.basic.ecliptic.par")
    m = get_model(parfile)
    assert "AstrometryEcliptic" in m.components


bad_trouble = ["J1923+2515_NANOGrav_9yv1.gls.par", "J1744-1134.basic.ecliptic.par"]


# @pytest.mark.xfail(reason="inexact conversions")
@pytest.mark.parametrize("parfile", glob(join(datadir, "*.par")))
def test_get_model_roundtrip(tmp_dir, parfile):
    if basename(parfile) in bad_trouble:
        pytest.skip("This parfile is unclear")
    try:
        m_old = get_model(parfile)
    except (ValueError, IOError, MissingParameter) as e:
        pytest.skip(f"Existing code raised an exception {e}")

    fn = join(tmp_dir, "file.par")
    with open(fn, "w") as f:
        f.write(m_old.as_parfile())
    m_roundtrip = get_model(fn)
    assert set(m_roundtrip.get_params_mapping().keys()) == set(
        m_old.get_params_mapping().keys()
    )
    assert set(m_roundtrip.components.keys()) == set(m_old.components.keys())

    # for p in m_old.get_params_mapping():
    #    assert getattr(m_old, p).quantity == getattr(m_roundtrip, p).quantity


def test_simple_manual():
    tm = TimingModel(
        name="test_manual", components=[AstrometryEquatorial(), Spindown()]
    )
    tm.setup()
    assert "F0" in tm.phase_deriv_funcs.keys()
    assert "RAJ" in tm.delay_deriv_funcs.keys()
    assert "DECJ" in tm.delay_deriv_funcs.keys()

    with pytest.raises(MissingParameter):  # No RA and DEC input
        tm.validate()

    tm.RAJ.value = "19:59:48"
    tm.DECJ.value = "20:48:36"
    tm.F0.quantity = 622.122030511927 * u.Hz
    tm.PEPOCH.value = 48196.0
    tm.validate()  # This should work.

    # When there is no POSEPOCH set, it should just be unset
    assert tm.POSEPOCH.value is None


def test_add_all_components():
    # models_by_category = defaultdict(list)
    # for k, c_type in Component.component_types.items():
    #     print(k, c_type)
    #     models_by_category[c_type.category].append(c_type)
    comps = [x() for x in Component.component_types.values()]
    tm = TimingModel(name="test_manual", components=comps)
    tm.setup()  # This should not have any parameter check.
    assert len(tm.components) == len(comps)
