import pytest
import numpy as np

from pint.models import get_model_and_toas
from pinttestdata import datadir


@pytest.fixture()
def model_and_toas():
    parfile = datadir / "J1713+0747_small.gls.par"
    timfile = datadir / "J1713+0747_small.tim"
    return get_model_and_toas(parfile, timfile)


def test_designmatrix_dims_no_free_noise_param(model_and_toas):
    """Parameters of the design matrix should be the free params and Offset.
    The dimensions of the design matrix should be Ntoas x (Nfree+1).
    """
    model, toas = model_and_toas

    # This should work.
    M, M_params, M_units = model.designmatrix(toas)

    assert set(M_params) == set(model.free_params + ["Offset"])
    assert np.all(M.shape == np.array([len(toas), len(model.free_params) + 1]))


@pytest.mark.parametrize("noiseparam", ["EFAC1", "EQUAD1", "ECORR1"])
def test_designmatrix_free_noise_params(model_and_toas, noiseparam):
    """Design matrix should ignore any unfrozen noise parameters."""
    model, toas = model_and_toas

    getattr(model, noiseparam).frozen = False

    # This should work and ignore the unfrozen noise parameter
    M, M_params, M_units = model.designmatrix(toas)

    assert noiseparam in model.free_params and noiseparam not in M_params

    # Reset the model.
    getattr(model, noiseparam).frozen = True
