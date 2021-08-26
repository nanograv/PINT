#! /usr/bin/env python
import os
import unittest
from os.path import join

import astropy.units as u
import pytest
from pinttestdata import datadir

import pint
from pint.fitter import (
    ConvergenceFailure,
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
        55000, 58000, 20, model=m, freq=1400 * u.MHz, dm=10 * pint.dmu
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
    try:
        dwls.fit_toas(maxiter=1)
    except MaxiterReached:
        pass

    assert abs(wls.resids.chi2 - dwls.resids.chi2) < 0.01


@pytest.mark.parametrize("full_cov", [False, True])
def test_compare_downhill_gls(full_cov, wls):
    gls = DownhillGLSFitter(wls.toas, wls.model_init)
    try:
        gls.fit_toas(maxiter=1, full_cov=full_cov)
    except MaxiterReached:
        pass

    # Why is this taking a different step from the plain GLS fitter?
    assert abs(wls.resids_init.chi2 - gls.resids_init.chi2) < 0.01
    assert abs(wls.resids.chi2 - gls.resids.chi2) < 0.01


@pytest.mark.parametrize("full_cov", [False, True])
def test_compare_downhill_wb(full_cov, wb):
    dwb = WidebandDownhillFitter(wb.toas, wb.model_init)
    try:
        dwb.fit_toas(maxiter=1, full_cov=full_cov)
    except MaxiterReached:
        pass

    assert abs(wb.resids.chi2 - dwb.resids.chi2) < 0.01
