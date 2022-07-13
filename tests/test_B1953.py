"""Various tests to assess the performance of the B1953+29."""
import warnings

import astropy.units as u
import numpy as np
import pytest
import test_derivative_utils as tdu
from astropy import log
from pinttestdata import datadir

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    s = Setup()

    s.parfileB1953 = datadir / "B1953+29_NANOGrav_dfg+12_TAI_FB90.par"
    s.timB1953 = datadir / "B1953+29_NANOGrav_dfg+12.tim"
    s.toasB1953 = toa.get_TOAs(
        s.timB1953,
        ephem="DE405",
        planets=False,
        include_bipm=False,
        picklefilename=pickle_dir,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        warnings.filterwarnings("ignore", message=r".*EPHVER.*")
        s.modelB1953 = mb.get_model(s.parfileB1953)
    # tempo result
    s.ltres, s.ltbindelay = np.genfromtxt(
        str(s.parfileB1953) + ".tempo2_test", skip_header=1, unpack=True
    )
    print(s.ltres)
    return s


def test_B1953_binary_delay(setup):
    # Calculate delays with PINT
    pint_binary_delay = setup.modelB1953.binarymodel_delay(setup.toasB1953, None)
    assert np.all(
        np.abs(pint_binary_delay.value + setup.ltbindelay) < 1e-8
    ), "B1953 binary delay test failed."


def test_B1953(setup):
    pint_resids_us = Residuals(
        setup.toasB1953, setup.modelB1953, use_weighted_mean=False
    ).time_resids.to(u.s)
    # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
    assert np.all(
        np.abs(pint_resids_us.value - setup.ltres) < 3e-8
    ), "B1953 residuals test failed."


def test_derivative(setup):
    log.setLevel("DEBUG")
    testp = tdu.get_derivative_params(setup.modelB1953)
    delay = setup.modelB1953.delay(setup.toasB1953)
    for p in testp.keys():
        log.debug("Runing derivative for {}".format("d_delay_d_" + p))
        ndf = setup.modelB1953.d_phase_d_param_num(setup.toasB1953, p, testp[p])
        adf = setup.modelB1953.d_phase_d_param(setup.toasB1953, delay, p)
        diff = adf - ndf
        if not np.all(diff.value) == 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            if p in ["ECC", "EDOT"]:
                tol = 20
            elif p in ["PMDEC"]:
                tol = 5e-3
            else:
                tol = 1e-3
            log.debug(
                "derivative relative diff for %s, %lf"
                % ("d_delay_d_" + p, np.nanmax(relative_diff).value)
            )
            assert np.nanmax(relative_diff) < tol, msg
        else:
            continue
