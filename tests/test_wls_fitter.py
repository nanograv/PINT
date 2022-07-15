#! /usr/bin/env python
import os

import pytest
from pinttestdata import datadir

from pint import toa
from pint.fitter import WLSFitter
from pint.models.model_builder import get_model

per_param = {
    "A1": 1e-05,
    "DECJ": 1e-06,
    "DMX_0003": 120,
    "ECC": 0.2,
    "F0": 1e-12,
    "F1": 0.001,
    "JUMP3": 10.0,
    "M2": 10.0,
    "OM": 1e-06,
    "PB": 1e-08,
    "PMDEC": 0.1,
    "PMRA": 0.1,
    "PX": 100,
    "RAJ": 1e-08,
    "SINI": -0.004075,
    "T0": 1e-10,
}


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    cls = Setup()
    cls.par = datadir / "B1855+09_NANOGrav_dfg+12_TAI_FB90.par"
    cls.tim = datadir / "B1855+09_NANOGrav_dfg+12.tim"
    cls.m = get_model(cls.par)
    cls.t = toa.get_TOAs(cls.tim, ephem="DE405", picklefilename=pickle_dir)
    cls.f = WLSFitter(cls.t, cls.m)
    # set perturb parameter step
    return cls


def perturb_param(setup, param, h):
    par = getattr(setup.f.model, param)
    orv = par.value
    par.value = (1 + h) * orv
    setup.f.model.free_params = [param]


@pytest.mark.parametrize("param, delta", list(per_param.items()))
def test_wls_fitter(setup, param, delta):
    perturb_param(setup, param, delta)
    setup.f.fit_toas()
    red_chi2 = setup.f.resids.reduced_chi2
    assert red_chi2 < 2.6


def test_has_correlated_errors(setup):
    assert not setup.f.resids.model.has_correlated_errors
