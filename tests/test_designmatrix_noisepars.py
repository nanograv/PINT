import os
import pint.models.model_builder as mb
import pinttestdata
import pytest


@pytest.fixture()
def model_and_toas():
    datadir = pinttestdata.datadir
    parfile = os.path.join(datadir, "J1713+0747_NANOGrav_11yv0_short.gls.par")
    timfile = os.path.join(datadir, "J1713+0747_NANOGrav_11yv0_short.tim")
    return mb.get_model_and_toas(parfile, timfile)


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
