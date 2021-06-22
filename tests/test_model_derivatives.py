import copy
import logging
import os.path
import warnings
from contextlib import contextmanager
from glob import glob

import numdifftools
import numpy as np
import pytest
from hypothesis import HealthCheck, Verbosity, assume, given, settings
from hypothesis.strategies import composite, floats, integers, sampled_from
from numpy.testing import assert_allclose

import pint.toa
from pint.models import get_model


@contextmanager
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        previous_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous_level)


class Values:
    def __repr__(self):
        return f"{os.path.basename(self.parfile)}::{self.param}"


_cached_models = {}


def get_model_cached(parfile):
    if parfile not in _cached_models:
        with quiet():
            _cached_models[parfile] = get_model(parfile)
    return copy.deepcopy(_cached_models[parfile])


_cached_toas = {}


def get_model_and_toas(parfile):
    if parfile not in _cached_toas:
        model = get_model_cached(parfile)
        with quiet():
            toas1 = pint.toa.make_fake_toas(
                model=model, startMJD=57000, endMJD=58000, ntoas=5, freq=1400, obs='gbt'
            )
            toas2 = pint.toa.make_fake_toas(
                model=model, startMJD=57100, endMJD=58000, ntoas=5, freq=2000, obs='ao'
            )
            toas = pint.toa.merge_TOAs([toas1, toas2])
            phase = model.phase(toas)
        _cached_toas[parfile] = model, toas, phase
    return copy.deepcopy(_cached_toas[parfile])


@pytest.mark.parametrize(
    "parfile, param",
    [
        ("J1955dd.par", param)
        for param in [
            "PX",
            "RAJ",
            "DECJ",
            "PMRA",
            "PMDEC",
            "F0",
            "F1",
            "PB",
            "A1",
            "ECC",  # NaNs for some reason
            "T0",
            "OM",
        ]
    ]
    + [
        ("J1853+1303_NANOGrav_11yv0.gls.par", param)  # ELL1H (?)
        for param in [
            "ELONG",
            "ELAT",
            "PMELONG",
            "PMELAT",
            "DMX_0027",
            "TASC",
            "EPS1",
            "EPS2",
            "H3",
            "PB",
            "A1",
            "A1DOT",  # Scaling weirdness
        ]
    ]
    + [("B1855+09_NANOGrav_12yv3.wb.gls.par", param) for param in ["M2", "SINI",]]
    + [
        ("J0023+0923_NANOGrav_11yv0.gls.par", param)
        for param in ["FB0", "FB1", "FB2", "FB3", "FD1", "JUMP1"]
    ]
    + [("J0613-0200_NANOGrav_9yv1_ELL1H.gls.par", param) for param in ["H3", "H4",]]
    + [
        ("J0613-0200_NANOGrav_9yv1_ELL1H_STIG.gls.par", param)
        for param in ["H3", "STIGMA",]
    ]
    + [
        ("J1713+0747_NANOGrav_11yv0.gls.par", param)  # DDK
        for param in ["PB", "A1", "ECC", "T0", "M2", "KIN", "KOM", "PX",]
    ],
)
def test_derivative_equals_numerical(parfile, param):
    model, toas, phase = get_model_and_toas(
        os.path.join(os.path.dirname(__file__), "datafile", parfile)
    )
    units = getattr(model, param).units

    def f(value):
        m = copy.deepcopy(model)
        getattr(m, param).value = value
        with quiet():
            warnings.simplefilter("ignore")
            try:
                dphase = m.phase(toas) - phase
            except ValueError:
                return np.nan * np.zeros_like(phase.frac)
        return dphase.int + dphase.frac

    if param == 'ECC':
        e = model.ECC.value
        stepgen = numdifftools.MaxStepGenerator(min(e, 1-e))
    else:
        stepgen = None
    df = numdifftools.Derivative(f, step=stepgen)

    assert_allclose(
        model.d_phase_d_param(toas, delay=None, param=param).to_value(1 / units),
        df(getattr(model, param).value),
        atol=1e-3,
        rtol=1e-3,
    )
