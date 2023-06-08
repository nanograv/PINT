import copy
import logging
import os.path
import warnings
from contextlib import contextmanager

import numdifftools
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u

import pint.toa
import pint.simulation
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
        if hasattr(model, "T0") and model.T0.value is not None:
            start = model.T0.value
        elif hasattr(model, "TASC") and model.TASC.value is not None:
            start = model.TASC.value
        elif hasattr(model, "PEPOCH") and model.PEPOCH.value is not None:
            start = model.PEPOCH.value
        else:
            start = 57000
        with quiet():
            toas1 = pint.simulation.make_fake_toas_uniform(
                model=model,
                startMJD=start,
                endMJD=start + 100,
                ntoas=5,
                freq=1400 * u.MHz,
                obs="gbt",
            )
            toas2 = pint.simulation.make_fake_toas_uniform(
                model=model,
                startMJD=start + 1,
                endMJD=start + 102,
                ntoas=5,
                freq=2000 * u.MHz,
                obs="ao",
            )
            toas = pint.toa.merge_TOAs([toas1, toas2])
            phase = model.phase(toas, abs_phase=False)
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
    + [("B1855+09_NANOGrav_12yv3.wb.gls.par", param) for param in ["M2", "SINI"]]
    + [
        ("J0023+0923_NANOGrav_11yv0.gls.par", param)
        for param in ["FB0", "FB1", "FB2", "FB3", "FD1", "JUMP1"]
    ]
    + [("J0613-0200_NANOGrav_9yv1_ELL1H.gls.par", param) for param in ["H3", "H4"]]
    + [
        ("J0613-0200_NANOGrav_9yv1_ELL1H_STIG.gls.par", param)
        for param in ["H3", "STIGMA"]
    ]
    + [
        (
            "J1713+0747_NANOGrav_11yv0.gls.par",
            param,
        )  # DDK; also A1DOT doesn't need rescaling
        for param in ["PB", "A1", "ECC", "T0", "M2", "KIN", "KOM", "PX", "A1DOT"]
    ],
)
def test_derivative_equals_numerical(parfile, param):
    if param == "H3":
        pytest.xfail("PINT's H3 code is known to use inconsistent approximations")
    model, toas, phase = get_model_and_toas(
        os.path.join(os.path.dirname(__file__), "datafile", parfile)
    )
    units = getattr(model, param).units

    def f(value):
        m = copy.deepcopy(model)
        p = getattr(m, param)
        if isinstance(p, pint.models.parameter.MJDParameter):
            p.value = value
        else:
            p.value = value * p.units
        with quiet():
            warnings.simplefilter("ignore")
            try:
                dphase = m.phase(toas, abs_phase=False) - phase
            except ValueError:
                return np.nan * np.zeros_like(phase.frac)
        return dphase.int + dphase.frac

    if param == "ECC":
        e = model.ECC.value
        stepgen = numdifftools.MaxStepGenerator(min(e, 1 - e) / 2)
    elif param == "H3":
        h3 = model.H3.value
        stepgen = numdifftools.MaxStepGenerator(abs(h3) / 2)
    elif param == "FB0":
        stepgen = numdifftools.MaxStepGenerator(np.abs(model.FB0.value) * 1e-2)
    elif param == "FB1":
        stepgen = numdifftools.MaxStepGenerator(np.abs(model.FB1.value) * 1e3)
    elif param == "FB2":
        stepgen = numdifftools.MaxStepGenerator(np.abs(model.FB2.value) * 1e5)
    elif param == "FB3":
        stepgen = numdifftools.MaxStepGenerator(np.abs(model.FB3.value) * 1e7)
    else:
        stepgen = None
    df = numdifftools.Derivative(f, step=stepgen)

    a = model.d_phase_d_param(toas, delay=None, param=param).to_value(1 / units)
    b = df(getattr(model, param).value)
    if param.startswith("FB"):
        assert np.amax(np.abs(a - b)) / np.amax(np.abs(a) + np.abs(b)) < 1.5e-6
    else:
        assert_allclose(a, b, atol=1e-4, rtol=1e-4)
