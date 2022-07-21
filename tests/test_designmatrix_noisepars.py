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


def _test_designmatrix(model, toas):
    try:
        # This should raise a NotImplementedError
        M = model.designmatrix(toas)
    except NotImplementedError:
        pass

    # This should work.
    M = model.designmatrix(toas, incnoise=False)


def test_designmatrix_noise_exception_falsepositive(model_and_toas):
    model, toas = model_and_toas
    _test_designmatrix(model, toas)


def test_designmatrix_noise_exception_efac(model_and_toas):
    model, toas = model_and_toas
    model.EFAC1.frozen = False
    _test_designmatrix(model, toas)


def test_designmatrix_noise_exception_equad(model_and_toas):
    model, toas = model_and_toas
    model.EQUAD1.frozen = False
    _test_designmatrix(model, toas)


def test_designmatrix_noise_exception_ecorr(model_and_toas):
    model, toas = model_and_toas
    model.ECORR1.frozen = False
    _test_designmatrix(model, toas)
