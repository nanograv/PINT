import os
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


def test_designmatrix_noise_exception_free_efac(model_and_toas):
    model, toas = model_and_toas
    model.EFAC1.frozen = False

    with pytest.raises(NotImplementedError):
        # This should raise a NotImplementedError rather than an AttributeError.
        M = model.designmatrix(toas)


def test_designmatrix_noise_exception_free_equad(model_and_toas):
    model, toas = model_and_toas
    model.EQUAD1.frozen = False

    with pytest.raises(NotImplementedError):
        # This should raise a NotImplementedError rather than an AttributeError.
        M = model.designmatrix(toas)


def test_designmatrix_noise_exception_ecorr(model_and_toas):
    model, toas = model_and_toas
    model.ECORR1.frozen = False

    with pytest.raises(NotImplementedError):
        # This should raise a NotImplementedError rather than an AttributeError.
        M = model.designmatrix(toas)
