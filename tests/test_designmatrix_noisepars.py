import pytest

from pint.models import get_model_and_toas
from pinttestdata import datadir


@pytest.fixture()
def model_and_toas():
    parfile = datadir / "J1713+0747_small.gls.par"
    timfile = datadir / "J1713+0747_small.tim"
    return get_model_and_toas(parfile, timfile)


def test_designmatrix_no_free_noise_param(model_and_toas):
    model, toas = model_and_toas

    # This should work.
    M = model.designmatrix(toas)


def test_designmatrix_free_noise_params(model_and_toas):
    model, toas = model_and_toas

    for noiseparam in ["EFAC1", "EQUAD1", "ECORR1"]:
        getattr(model, noiseparam).frozen = False

        # This should work and ignore the unfrozen noise parameter
        M, M_params, M_units = model.designmatrix(toas)

        assert noiseparam in model.free_params and noiseparam not in M_params

        # Reset the model.
        getattr(model, noiseparam).frozen = True
