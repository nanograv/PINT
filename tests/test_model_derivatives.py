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


models = {}
bad_models = set()


@composite
def model_and_free_param(draw):
    known_bad = set()
    parfiles = sorted(
        set(
            glob(os.path.join(os.path.dirname(__file__), "datafile", "*.par"))
        ).difference(known_bad)
    )
    parfile = draw(sampled_from(parfiles))
    try:
        m = models[parfile]
    except KeyError:
        assume(parfile not in bad_models)
        with quiet():
            try:
                m = get_model(parfile)
            except (ValueError, AttributeError):
                bad_models.add(parfile)
                assume(False)
            else:
                models[parfile] = m
    assume(m.free_params)
    v = Values()
    v.parfile = parfile
    v.model = m
    v.param = draw(sampled_from(m.free_params))
    return v


# @settings(suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.verbose)
# @given(model_and_free_param())
# def test_load_ok(model_and_param):
#    pass


# For some reason this is extremely slow
@settings(suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.verbose)
# @settings(suppress_health_check=[HealthCheck.too_slow])
@given(model_and_free_param(), floats(50000, 60000))
def disabled_test_derivative_equals_numerical(model_and_param, start):
    _, model, param = (
        model_and_param.parfile,
        model_and_param.model,
        model_and_param.param,
    )
    with quiet():
        toas = pint.toa.make_fake_toas(
            model=model, startMJD=start, endMJD=start + 1, ntoas=2
        )
        phase = model.phase(toas)
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

    df = numdifftools.Derivative(f)

    assert_allclose(
        model.d_phase_d_param(toas, delay=None, param=param).to_value(1 / units),
        df(getattr(model, param).value),
        atol=1e-3,
        rtol=1e-3,
    )


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
            toas = pint.toa.make_fake_toas(
                model=model, startMJD=57000, endMJD=58000, ntoas=5
            )
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
        for param in ["FB0", "FB1", "FB2", "FB3",]
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
def test_derivative_equals_numerical_multi(parfile, param):
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

    df = numdifftools.Derivative(f)

    assert_allclose(
        model.d_phase_d_param(toas, delay=None, param=param).to_value(1 / units),
        df(getattr(model, param).value),
        atol=1e-3,
        rtol=1e-3,
    )
