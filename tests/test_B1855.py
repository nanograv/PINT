"""Various tests to assess the performance of the B1855+09."""
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
    s.parfileB1855 = datadir / "B1855+09_NANOGrav_dfg+12_TAI_FB90.par"
    s.timB1855 = datadir / "B1855+09_NANOGrav_dfg+12.tim"
    s.toasB1855 = toa.get_TOAs(
        s.timB1855,
        ephem="DE405",
        planets=False,
        include_bipm=False,
        picklefilename=pickle_dir,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        warnings.filterwarnings("ignore", message=r".*EPHVER.*")
        s.modelB1855 = mb.get_model(s.parfileB1855)
    logging.debug("%s" % s.modelB1855.components)
    logging.debug("%s" % s.modelB1855.params)
    # tempo result
    s.ltres = np.genfromtxt(
        str(s.parfileB1855) + ".tempo2_test", skip_header=1, unpack=True
    )
    return s


def test_B1855(setup):
    pint_resids_us = Residuals(
        setup.toasB1855, setup.modelB1855, use_weighted_mean=False
    ).time_resids.to(u.s)
    # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
    assert np.all(
        np.abs(pint_resids_us.value - setup.ltres) < 3e-8
    ), "B1855 residuals test failed."


def test_derivative(setup):
    log = logging.getLogger("TestB1855.derivative_test")
    testp = tdu.get_derivative_params(setup.modelB1855)
    delay = setup.modelB1855.delay(setup.toasB1855)
    for p in testp.keys():
        log.debug("Runing derivative for %s", "d_delay_d_" + p)
        ndf = setup.modelB1855.d_phase_d_param_num(setup.toasB1855, p, testp[p])
        adf = setup.modelB1855.d_phase_d_param(setup.toasB1855, delay, p)
        diff = adf - ndf
        if not np.all(diff.value) == 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            if p in ["SINI"]:
                tol = 0.7
            else:
                tol = 1e-3
            log.debug(
                "derivative relative diff for %s, %lf"
                % ("d_delay_d_" + p, np.nanmax(relative_diff).value)
            )
            assert np.nanmax(relative_diff) < tol, msg
        else:
            continue
