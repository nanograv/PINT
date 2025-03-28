import pytest
from pint.models import get_model, get_model_and_toas
from pint.simulation import make_fake_toas_uniform
from pint.fitter import DownhillWLSFitter, DownhillGLSFitter
from pinttestdata import datadir


from io import StringIO
import numpy as np


@pytest.fixture
def data_1():
    par = """
        ELAT    1.3     1
        ELONG   2.5     1
        F0      100     1
        F1      1e-13   1
        PEPOCH  55000
        EPHEM   DE440
        EFAC mjd 50000 53000 2      1
        EQUAD mjd 53000 55000 0.8    1
    """

    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(50000, 55000, 200, m, add_noise=True)

    return m, t


@pytest.fixture
def data_2():
    return get_model_and_toas(
        datadir / "ecorr_fit_test.par", datadir / "ecorr_fit_test.tim"
    )


@pytest.fixture
def data_3():
    par = """
        ELAT    1.3     1
        ELONG   2.5     1
        F0      100     1
        F1      1e-13   1
        PEPOCH  55000
        EPHEM   DE440
        EFAC mjd 50000 53000 2      1
        EQUAD mjd 53000 55000 0.8    1
        TNREDAMP    -13  
        TNREDGAM    3
        TNREDC      5
    """

    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(
        50000, 55000, 200, m, add_noise=True, add_correlated_noise=True
    )

    return m, t


def test_white_noise_fit(data_1):
    m, t = data_1

    assert m.EFAC1.uncertainty_value == 0 and m.EQUAD1.uncertainty_value == 0

    ftr = DownhillWLSFitter(t, m)
    ftr.fit_toas(maxiter=5)

    assert (
        ftr.model.EFAC1.uncertainty_value > 0 and ftr.model.EQUAD1.uncertainty_value > 0
    )
    assert (
        np.abs(m.EFAC1.value - ftr.model.EFAC1.value)
        / ftr.model.EFAC1.uncertainty_value
        < 4
    )
    assert (
        np.abs(m.EQUAD1.value - ftr.model.EQUAD1.value)
        / ftr.model.EQUAD1.uncertainty_value
        < 4
    )


def test_white_noise_refit(data_1):
    m, t = data_1

    ftr = DownhillWLSFitter(t, m)

    ftr.model.EFAC1.value = 1.5
    ftr.model.EQUAD1.value = 0.5

    ftr.fit_toas(maxiter=5)

    assert (
        np.abs(m.EFAC1.value - ftr.model.EFAC1.value)
        / ftr.model.EFAC1.uncertainty_value
        < 4
    )
    assert (
        np.abs(m.EQUAD1.value - ftr.model.EQUAD1.value)
        / ftr.model.EQUAD1.uncertainty_value
        < 4
    )


def test_ecorr_fit(data_2):
    m2, t2 = data_2

    ftr = DownhillGLSFitter(t2, m2)
    ftr.fit_toas()

    assert ftr.model.ECORR1.uncertainty_value > 0
    assert (
        np.abs(m2.ECORR1.value - ftr.model.ECORR1.value)
        / ftr.model.ECORR1.uncertainty_value
        < 4
    )


def test_ecorr_refit(data_2):
    m2, t2 = data_2

    ftr = DownhillGLSFitter(t2, m2)

    ftr.model.ECORR1.value = 0.75

    ftr.fit_toas()

    assert ftr.model.ECORR1.uncertainty_value > 0
    assert (
        np.abs(m2.ECORR1.value - ftr.model.ECORR1.value)
        / ftr.model.ECORR1.uncertainty_value
        < 4
    )


def test_white_noise_fit_gls(data_3):
    m, t = data_3

    assert m.EFAC1.uncertainty_value == 0 and m.EQUAD1.uncertainty_value == 0

    ftr = DownhillGLSFitter(t, m)
    ftr.fit_toas(maxiter=5)

    assert (
        ftr.model.EFAC1.uncertainty_value > 0 and ftr.model.EQUAD1.uncertainty_value > 0
    )
    assert (
        np.abs(m.EFAC1.value - ftr.model.EFAC1.value)
        / ftr.model.EFAC1.uncertainty_value
        < 4
    )
    assert (
        np.abs(m.EQUAD1.value - ftr.model.EQUAD1.value)
        / ftr.model.EQUAD1.uncertainty_value
        < 4
    )

    assert np.all(np.isfinite(ftr.resids.calc_whitened_resids()))
