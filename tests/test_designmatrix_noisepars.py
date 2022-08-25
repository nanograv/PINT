import os
import pytest

from pint.models import get_model_and_toas
from pinttestdata import datadir


@pytest.fixture()
def model_and_toas():
    parfile = os.path.join(datadir, "J1713+0747_small.gls.par")
    timfile = os.path.join(datadir, "J1713+0747_small.tim")
    return get_model_and_toas(parfile, timfile)


def test_designmatrix_noise_exception_no_free_noise_param(model_and_toas):
    model, toas = model_and_toas

    # Both of these should work.
    M = model.designmatrix(toas)
    M = model.designmatrix(toas, incnoise=False)


def test_designmatrix_noise_exception_free_efac(model_and_toas):
    model, toas = model_and_toas
    model.EFAC1.frozen = False

    with pytest.raises(NotImplementedError):
        # This should raise a NotImplementedError rather than an AttributeError.
        M = model.designmatrix(toas)

    # This should work.
    M = model.designmatrix(toas, incnoise=False)

    model.EFAC1.frozen = True


def test_designmatrix_noise_exception_free_equad(model_and_toas):
    model, toas = model_and_toas
    model.EQUAD1.frozen = False

    with pytest.raises(NotImplementedError):
        # This should raise a NotImplementedError rather than an AttributeError.
        M = model.designmatrix(toas)

    # This should work.
    M = model.designmatrix(toas, incnoise=False)

    model.EQUAD1.frozen = True


def test_designmatrix_noise_exception_ecorr(model_and_toas):
    model, toas = model_and_toas
    model.ECORR1.frozen = False

    with pytest.raises(NotImplementedError):
        # This should raise a NotImplementedError rather than an AttributeError.
        M = model.designmatrix(toas)

    # This should work.
    M = model.designmatrix(toas, incnoise=False)

    model.ECORR1.frozen = True
