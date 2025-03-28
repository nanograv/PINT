import contextlib
from os.path import join
from io import StringIO
import copy
import numpy as np

import astropy.units as u
import pytest
from pinttestdata import datadir

from pint.fitter import (
    MaxiterReached,
    DownhillGLSFitter,
    DownhillWLSFitter,
    GLSFitter,
    WidebandDownhillFitter,
    WidebandTOAFitter,
    WLSFitter,
)
from pint.models.model_builder import get_model
from pint.toa import get_TOAs
from pint.simulation import make_fake_toas_uniform


@pytest.fixture
def wls():
    m = get_model(join(datadir, "NGC6440E.par"))
    t = get_TOAs(join(datadir, "NGC6440E.tim"), ephem="DE421")

    wls = WLSFitter(t, m)
    wls.fit_toas()

    return wls


@pytest.fixture
def wb():
    m = get_model(join(datadir, "NGC6440E.par"))
    t = make_fake_toas_uniform(
        55000, 58000, 20, model=m, freq=1400 * u.MHz, wideband=True
    )

    wb = WidebandTOAFitter(t, m)
    wb.fit_toas()

    return wb


@pytest.mark.parametrize("full_cov", [False, True])
def test_compare_gls(full_cov, wls):
    gls = GLSFitter(wls.toas, wls.model_init)
    gls.fit_toas(full_cov=full_cov)

    assert abs(wls.resids.chi2 - gls.resids.chi2) < 0.01


def test_compare_downhill_wls(wls):
    dwls = DownhillWLSFitter(wls.toas, wls.model_init)
    with contextlib.suppress(MaxiterReached):
        dwls.fit_toas(maxiter=1)
    assert abs(wls.resids.chi2 - dwls.resids.chi2) < 0.01


@pytest.mark.parametrize("full_cov", [False, True])
def test_compare_downhill_gls(full_cov, wls):
    gls = DownhillGLSFitter(wls.toas, wls.model_init)
    with contextlib.suppress(MaxiterReached):
        gls.fit_toas(maxiter=1, full_cov=full_cov)
    # Why is this taking a different step from the plain GLS fitter?
    assert abs(wls.resids_init.chi2 - gls.resids_init.chi2) < 0.01
    assert abs(wls.resids.chi2 - gls.resids.chi2) < 0.01


@pytest.mark.parametrize("full_cov", [False, True])
def test_compare_downhill_wb(full_cov, wb):
    dwb = WidebandDownhillFitter(wb.toas, wb.model_init)
    with contextlib.suppress(MaxiterReached):
        dwb.fit_toas(maxiter=1, full_cov=full_cov)
    assert abs(wb.resids.chi2 - dwb.resids.chi2) < 0.01


@pytest.fixture
def m_t():
    model = get_model(
        StringIO(
            """
                PSR J1234+5678
                ELAT 0
                ELONG 0
                F0 1 1
                F1 0 1
                DM 10
                PEPOCH 56000
                EFAC mjd 55000 56000 1
                EFAC mjd 56000 57000 2
            """
        )
    )
    toas = make_fake_toas_uniform(
        55000, 57000, 20, model=model, add_noise=True, wideband=True
    )
    return model, toas


@pytest.mark.parametrize(
    "fitter",
    [
        WLSFitter,
        DownhillWLSFitter,
        GLSFitter,
        DownhillGLSFitter,
        WidebandTOAFitter,
        WidebandDownhillFitter,
    ],
)
def test_step_different_with_efacs(fitter, m_t):
    m, t = m_t
    f = fitter(t, m)
    with contextlib.suppress(MaxiterReached):
        f.fit_toas(maxiter=1)
    m2 = copy.deepcopy(m)
    m2.EFAC1.value = 1
    m2.EFAC2.value = 1
    f2 = fitter(t, m2)
    with contextlib.suppress(MaxiterReached):
        f2.fit_toas(maxiter=1)
    for p in m.free_params:
        assert getattr(f.model, p).value != getattr(f2.model, p).value


@pytest.mark.parametrize(
    "fitter",
    [
        GLSFitter,
        DownhillGLSFitter,
        WidebandTOAFitter,
        WidebandDownhillFitter,
    ],
)
def test_step_different_with_efacs_full_cov(fitter, m_t):
    m, t = m_t
    f = fitter(t, m)
    with contextlib.suppress(MaxiterReached):
        f.fit_toas(maxiter=1, full_cov=True)
    m2 = copy.deepcopy(m)
    m2.EFAC1.value = 1
    m2.EFAC2.value = 1
    f2 = fitter(t, m2)
    with contextlib.suppress(MaxiterReached):
        f2.fit_toas(maxiter=1, full_cov=True)
    for p in m.free_params:
        assert getattr(f.model, p).value != getattr(f2.model, p).value


@pytest.mark.parametrize(
    "fitter1, fitter2",
    [
        (WLSFitter, DownhillWLSFitter),
        (GLSFitter, DownhillGLSFitter),
        (WidebandTOAFitter, WidebandDownhillFitter),
    ],
)
def test_downhill_same_step(fitter1, fitter2, m_t):
    m, t = m_t
    f1 = fitter1(t, m)
    f2 = fitter2(t, m)
    with contextlib.suppress(MaxiterReached):
        f1.fit_toas(maxiter=1)
    with contextlib.suppress(MaxiterReached):
        f2.fit_toas(maxiter=1)
    for p in m.free_params:
        assert np.isclose(
            getattr(f1.model, p).value - getattr(f1.model_init, p).value,
            getattr(f2.model, p).value - getattr(f2.model_init, p).value,
        )
