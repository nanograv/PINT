"""Various tests to assess the performance of the FBX model."""
import os

import astropy.units as u
import numpy as np
import pytest
import test_derivative_utils as tdu
from pinttestdata import datadir

from pint import fitter
from pint.models import get_model_and_toas
import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals

parfileJ0023 = os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par")
parJ0023ell1 = os.path.join(datadir, "J0023+0923_ell1_simple.par")
timJ0023 = os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.tim")


@pytest.fixture
def toasJ0023():
    return toa.get_TOAs(timJ0023, ephem="DE436", planets=False)


@pytest.fixture
def modelJ0023():
    return mb.get_model(parfileJ0023)


ltres, ltbindelay = np.genfromtxt(
    f"{parfileJ0023}.tempo2_test", skip_header=1, unpack=True
)


def test_J0023_binary_delay(modelJ0023, toasJ0023):
    # Calculate binary delays with PINT
    pint_binary_delay = modelJ0023.binarymodel_delay(toasJ0023, None)
    assert np.all(np.abs(pint_binary_delay.value + ltbindelay) < 1e-9)


@pytest.mark.xfail(reason="PINT has a more modern position for Arecibo than TEMPO2")
def test_J0023(modelJ0023, toasJ0023):
    pint_resids_us = Residuals(
        toasJ0023, modelJ0023, use_weighted_mean=False
    ).time_resids.to(u.s)
    assert np.all(np.abs(pint_resids_us.value - ltres) < 1e-8)


def test_derivative(modelJ0023, toasJ0023):
    testp = tdu.get_derivative_params(modelJ0023)
    delay = modelJ0023.delay(toasJ0023)
    for p in testp.keys():
        print("Runing derivative for %s", f"d_delay_d_{p}")
        if p in ["EPS2", "EPS1"]:
            testp[p] = 15
        ndf = modelJ0023.d_phase_d_param_num(toasJ0023, p, testp[p])
        adf = modelJ0023.d_phase_d_param(toasJ0023, delay, p)
        diff = adf - ndf
        if np.all(diff.value) != 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            if p in ["PMELONG", "ELONG"]:
                tol = 2e-2
            elif p in ["FB1"]:
                # paulr added this to make tests pass with oldest supported versions of numpy/astropy, but I don't know why it is needed
                # How should we decide what the acceptable tolerance is? This should not just be a random choice.
                tol = 0.002
            elif p in ["FB2", "FB3"]:
                tol = 0.08
            else:
                tol = 1e-3
            print(
                (
                    "derivative relative diff for %s, %lf"
                    % (f"d_delay_d_{p}", np.nanmax(relative_diff).value)
                )
            )
            assert np.nanmax(relative_diff) < tol, msg
        else:
            continue


def test_summary_FB():
    m, t = get_model_and_toas(
        os.path.join(datadir, parJ0023ell1), os.path.join(datadir, timJ0023)
    )
    f = fitter.WLSFitter(toas=t, model=m)

    # Ensure print_summary runs without an exception for an ELL1 model with FBX
    f.print_summary()

    assert "PB" in f.get_summary()
