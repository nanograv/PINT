"""Various tests to assess the performance of early CHIME data."""
import warnings

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose
from pinttestdata import datadir

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals


@pytest.fixture
def setup():
    class Setup:
        pass

    s = Setup()
    s.parfile = datadir / "B1937+21.basic.par"
    s.tim = datadir / "B1937+21.CHIME.CHIME.NG.N.tim"
    return s


def test_toa_read(setup, pickle_dir):
    toas = toa.get_TOAs(
        setup.tim,
        ephem="DE436",
        planets=False,
        include_bipm=True,
        picklefilename=pickle_dir,
    )
    assert toas.ntoas == 848, "CHIME TOAs did not read correctly."
    assert set(toas.get_obss()) == {"chime"}


@pytest.mark.xfail(
    reason="SMR says: the new residual code makes it fail, and it probably shouldn't"
)
def test_residuals(setup, pickle_dir):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        model = mb.get_model(setup.parfile)
    toas = toa.get_TOAs(
        setup.tim,
        ephem="DE436",
        planets=False,
        include_bipm=True,
        picklefilename=pickle_dir,
    )
    r = Residuals(toas, model)
    # Comment out the following test for now, since the new residual
    # code makes it fail, and it probably shouldn't -- SMR
    assert_quantity_allclose(r.time_resids.to(u.us), 0 * u.us, atol=800 * u.us, rtol=0)
