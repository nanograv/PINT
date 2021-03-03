#! /usr/bin/env python
import os
import unittest
from os.path import join

import pytest
from pinttestdata import datadir

from pint.toa import get_TOAs
from pint.fitter import GLSFitter, WLSFitter, DownhillWLSFitter, DownhillGLSFitter
from pint.models.model_builder import get_model


@pytest.fixture
def wls():
    m = get_model(join(datadir, "NGC6440E.par"))
    t = get_TOAs(join(datadir, "NGC6440E.tim"), ephem="DE421")

    wls = WLSFitter(t, m)
    wls.fit_toas()

    return wls


@pytest.mark.parametrize("full_cov", [False, True])
def test_compare_gls(full_cov, wls):
    gls = GLSFitter(wls.toas, wls.model_init)
    gls.fit_toas(full_cov=full_cov)

    assert abs(wls.resids.chi2 - gls.resids.chi2) < 0.01


def test_compare_downhill_gls(wls):
    dwls = DownhillWLSFitter(wls.toas, wls.model_init)
    dwls.fit_toas()

    assert abs(wls.resids.chi2 - dwls.resids.chi2) < 0.01


@pytest.mark.parametrize("full_cov", [False, True])
def test_compare_downhill_gls(full_cov, wls):
    gls = DownhillGLSFitter(wls.toas, wls.model_init)
    gls.fit_toas(full_cov=full_cov)

    assert abs(wls.resids.chi2 - gls.resids.chi2) < 0.01
