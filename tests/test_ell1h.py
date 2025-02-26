"""Tests of ELL1H model """

import logging
import os
import pytest

import astropy.units as u
import numpy as np

import pint.fitter as ff
from pint.models import get_model
import pint.toa as toa
import test_derivative_utils as tdu
from pint.residuals import Residuals
from pinttestdata import datadir
from io import StringIO


os.chdir(datadir)


simple_par = """
      PSR    J0613-0200
      LAMBDA 93.7990065496191  1 0.0000000158550
      BETA   -25.4071326875232  1 0.0000000369013
      PMLAMBDA 2.1192  1 0.0174
      PMBETA -10.3422  1              0.0433
      PX 0.9074  1              0.1509
      F0 326.6005670972169810  1  0.0000000000066373
      F1 -1.022985317101D-15  1  6.219122230955D-20
      PEPOCH        54890.000000
      DM         38.778683
      BINARY ELL1H
      A1 1.091442190  1  0.000000598
      PB 1.19851255667964  1 0.00000000001332
      TASC 54889.991808565  1 0.000000012
      EPS1 0.0000025554  1 0.0000002783
      EPS2 0.0000036160  1 0.0000000847
      H3 2.7507208E-7  1       1.5114416E-7
      H4 2.0262048E-7  1       1.1276173E-7
      """


@pytest.fixture(scope="module")
def toasJ0613():
    return toa.get_TOAs("J0613-0200_NANOGrav_9yv1.tim", ephem="DE421", planets=False)


@pytest.fixture(scope="module")
def toasJ1853():
    return toa.get_TOAs("J1853+1303_NANOGrav_11yv0.tim", ephem="DE421", planets=False)


@pytest.fixture
def modelJ0613():
    return get_model("J0613-0200_NANOGrav_9yv1_ELL1H.gls.par")


@pytest.fixture
def modelJ1853():
    return get_model("J1853+1303_NANOGrav_11yv0.gls.par")


@pytest.fixture()
def modelJ0613_STIG():
    return get_model("J0613-0200_NANOGrav_9yv1_ELL1H_STIG.gls.par")


@pytest.fixture()
def tempo2_res():
    parfileJ1853 = "J1853+1303_NANOGrav_11yv0.gls.par"
    return np.genfromtxt(f"{parfileJ1853}.tempo2_test", skip_header=1, unpack=True)


def test_J1853(toasJ1853, modelJ1853, tempo2_res):
    """Test J1853 residuals with TEMPO2"""
    pint_resids_us = Residuals(
        toasJ1853, modelJ1853, use_weighted_mean=False
    ).time_resids.to(u.s)
    # Due to PINT has higher order of ELL1 model, Tempo2 gives a difference around 3e-8
    # Changed to 4e-8 since modification to get_PSR_freq() makes this 3.1e-8
    log = logging.getLogger("TestJ1853.J1853_residuals")
    diffs = np.abs(pint_resids_us.value - tempo2_res[0])
    log.debug("Diffs: %s\nMax: %s" % (diffs, np.max(diffs)))
    assert np.all(diffs < 4e-8), "J1853 residuals test failed."


def test_J1853_binary_delay(toasJ1853, modelJ1853, tempo2_res):
    # Calculate delays with PINT
    pint_binary_delay = modelJ1853.binarymodel_delay(toasJ1853, None)
    assert np.all(
        np.abs(pint_binary_delay.value + tempo2_res[1]) < 3e-8
    ), "J1853 binary delay test failed."


# TODO need a better derivative test
def test_derivative(toasJ1853, modelJ1853):
    log = logging.getLogger("TestJ1853.derivative_test")
    # only H3 is set in this model, so the other derivatives don't make sense
    test_params = ["H3"]  # , "H4", "STIGMA"]
    modelJ1853.H4.value = 0.0  # For test PBDOT
    modelJ1853.STIGMA.value = 0.0
    testp = tdu.get_derivative_params(modelJ1853)
    delay = modelJ1853.delay(toasJ1853)
    # Change parameter test step
    testp["H3"] = 6e-1
    #    testp["H4"] = 1e-2
    #    testp["STIGMA"] = 1e-2
    for p in test_params:
        log.debug(f"Runing derivative for {p}: d_delay_d_{p}")
        ndf = modelJ1853.d_phase_d_param_num(toasJ1853, p, testp[p])
        adf = modelJ1853.d_phase_d_param(toasJ1853, delay, p)
        diff = adf - ndf
        if np.all(diff.value) != 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            tol = 0.05 if p in ["EPS1DOT", "EPS1"] else 1e-3
            log.debug(
                (
                    "derivative relative diff for %s, %lf"
                    % (f"d_delay_d_{p}", np.nanmax(relative_diff).value)
                )
            )
            assert np.nanmax(relative_diff) < tol, msg
        else:
            continue


def test_J0613_fit_with_H3_H4(toasJ0613, modelJ0613):
    log = logging.getLogger("TestJ0613.fit_tests")
    f = ff.GLSFitter(toasJ0613, modelJ0613)
    f.fit_toas()


def test_J0613_STIG(toasJ0613, modelJ0613_STIG):
    log = logging.getLogger("TestJ0613.fit_tests_stig")
    f = ff.GLSFitter(toasJ0613, modelJ0613_STIG)
    f.fit_toas()


def test_SINI_raises():
    """SINI is not a valid parameter for ELL1H"""
    SINI_par = simple_par.replace("H3 2.7507208E-7", "SINI 0.8")
    with pytest.raises(ValueError):
        get_model(StringIO(SINI_par))


def test_M2_raises():
    """M2 is not a valid parameter for ELL1H"""
    M2_par = simple_par + "\nM2 1.0 1 0.1"
    with pytest.raises(AttributeError):
        get_model(StringIO(M2_par))


def test_no_H3_H4(toasJ0613):
    """Test no H3 and H4 in model."""
    no_H3_H4 = simple_par.replace("H4 2.0262048E-7  1       1.1276173E-7", "")
    no_H3_H4 = no_H3_H4.replace("H3 2.7507208E-7  1       1.5114416E-7", "")
    no_H3_H4_model = get_model(StringIO(no_H3_H4))
    assert no_H3_H4_model.H3.value is None
    assert no_H3_H4_model.H4.value is None
    test_toas = toasJ0613[::20]
    f = ff.WLSFitter(test_toas, no_H3_H4_model)
    f.fit_toas()


def test_H3_and_H4_non_zero(toasJ0613):
    """Testing if the different H3, H4 combination breaks the fitting. the
    fitting result will not be checked here.
    """
    simple_model = get_model(StringIO(simple_par))

    test_toas = toasJ0613[::20]
    f = ff.WLSFitter(test_toas, simple_model)
    f.fit_toas()


def test_zero_H4(toasJ0613):
    H4_zero_model = get_model(StringIO(simple_par))
    H4_zero_model.H4.value = 0.0
    H4_zero_model.H4.frozen = False
    assert H4_zero_model.H3.value != 0.0
    H4_zero_model.H3.frozen = False
    test_toas = toasJ0613[::20]
    f = ff.WLSFitter(test_toas, H4_zero_model)
    f.fit_toas()


def test_zero_H3(toasJ0613):
    H3_zero_model = get_model(StringIO(simple_par))
    H3_zero_model.H3.value = 0.0
    H3_zero_model.H3.frozen = False
    assert H3_zero_model.H4.value != 0.0
    H3_zero_model.H4.frozen = False
    test_toas = toasJ0613[::20]
    with pytest.raises(ValueError):
        ff.WLSFitter(test_toas, H3_zero_model)


def test_zero_H3_H4_fit_H3_H4(toasJ0613):
    H3H4_zero_model = get_model(StringIO(simple_par))
    H3H4_zero_model.H3.value = 0.0
    H3H4_zero_model.H4.value = 0.0
    H3H4_zero_model.H3.frozen = False
    H3H4_zero_model.H4.frozen = False
    # Make sure H3's derivative does not return zero.
    test_toas = toasJ0613[::20]
    d_h3 = H3H4_zero_model.d_delay_d_param(test_toas, "H3")
    assert d_h3.mean().value != 0.0
    assert d_h3.std().value != 0.0
    f = ff.WLSFitter(test_toas, H3H4_zero_model)
    f.fit_toas()
    assert f.model.H3.value > 0.0


def test_zero_H3_H4_fit_H3(toasJ0613):
    H3H4_zero2_model = get_model(StringIO(simple_par))
    H3H4_zero2_model.H3.value = 0.0
    H3H4_zero2_model.H3.frozen = False
    H3H4_zero2_model.H4.value = 0.0
    H3H4_zero2_model.H4.frozen = True
    test_toas = toasJ0613[::20]
    f = ff.WLSFitter(test_toas, H3H4_zero2_model)
    # This should work
    f.fit_toas()
    assert f.model.H3.value > 0.0
