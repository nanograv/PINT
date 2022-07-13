"""Various tests to assess the performance of the J0623-0200."""
import logging
import warnings

import astropy.units as u
import numpy as np
import pytest
import test_derivative_utils as tdu
from pinttestdata import datadir

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    s = Setup()
    s.parfileJ0613 = datadir / "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par"
    s.timJ0613 = datadir / "J0613-0200_NANOGrav_dfg+12.tim"
    s.toasJ0613 = toa.get_TOAs(
        s.timJ0613,
        ephem="DE405",
        planets=False,
        include_bipm=False,
        picklefilename=pickle_dir,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        warnings.filterwarnings("ignore", message=r".*EPHVER.*")
        s.modelJ0613 = mb.get_model(s.parfileJ0613)
    # tempo result
    s.ltres, s.ltbindelay = np.genfromtxt(
        str(s.parfileJ0613) + ".tempo2_test", skip_header=1, unpack=True
    )
    print(s.ltres)
    return s


def test_J0613_binary_delay(setup):
    # Calculate delays with PINT
    pint_binary_delay = setup.modelJ0613.binarymodel_delay(setup.toasJ0613, None)
    assert np.all(
        np.abs(pint_binary_delay.value + setup.ltbindelay) < 1e-8
    ), "J0613 binary delay test failed."


def test_J0613(setup):
    pint_resids_us = Residuals(
        setup.toasJ0613, setup.modelJ0613, use_weighted_mean=False
    ).time_resids.to(u.s)
    # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
    assert np.all(
        np.abs(pint_resids_us.value - setup.ltres) < 3e-8
    ), "J0613 residuals test failed."


def test_derivative(setup):
    log = logging.getLogger("TestJ0613.derivative_test")
    setup.modelJ0613.PBDOT.value = 0.0  # For test PBDOT
    setup.modelJ0613.EPS1DOT.value = 0.0
    setup.modelJ0613.EPS2DOT.value = 0.0
    setup.modelJ0613.A1DOT.value = 0.0
    testp = tdu.get_derivative_params(setup.modelJ0613)
    delay = setup.modelJ0613.delay(setup.toasJ0613)
    # Change parameter test step
    testp["EPS1"] = 1
    testp["EPS2"] = 1
    testp["PMDEC"] = 1
    testp["PMRA"] = 1
    for p in testp.keys():
        log.debug("Runing derivative for %s", "d_delay_d_" + p)
        ndf = setup.modelJ0613.d_phase_d_param_num(setup.toasJ0613, p, testp[p])
        adf = setup.modelJ0613.d_phase_d_param(setup.toasJ0613, delay, p)
        diff = adf - ndf
        if not np.all(diff.value) == 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            if p in ["EPS1DOT", "EPS1"]:
                tol = 0.05
            else:
                tol = 1e-3
            log.debug(
                "derivative relative diff for %s, %lf"
                % ("d_delay_d_" + p, np.nanmax(relative_diff).value)
            )
            assert np.nanmax(relative_diff) < tol, msg
        else:
            continue
