"""Tests of ELL1H model """
import logging
import os
import unittest

import astropy.units as u
import numpy as np

import pint.fitter as ff
import pint.models as model
import pint.toa as toa
import test_derivative_utils as tdu
from pint.residuals import Residuals
from pinttestdata import datadir


class TestELL1H(unittest.TestCase):
    """Compare delays from the ELL1 model with tempo and PINT"""

    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parfileJ1853 = "J1853+1303_NANOGrav_11yv0.gls.par"
        cls.timfileJ1853 = "J1853+1303_NANOGrav_11yv0.tim"
        cls.toasJ1853 = toa.get_TOAs(cls.timfileJ1853, ephem="DE421", planets=False)
        cls.modelJ1853 = model.get_model(cls.parfileJ1853)
        cls.ltres, cls.ltbindelay = np.genfromtxt(
            cls.parfileJ1853 + ".tempo2_test", skip_header=1, unpack=True
        )

        cls.parfileJ0613 = "J0613-0200_NANOGrav_9yv1_ELL1H.gls.par"
        cls.timfileJ0613 = "J0613-0200_NANOGrav_9yv1.tim"
        cls.modelJ0613 = model.get_model(cls.parfileJ0613)
        cls.toasJ0613 = toa.get_TOAs(cls.timfileJ0613, ephem="DE421", planets=False)
        cls.parfileJ0613_STIG = "J0613-0200_NANOGrav_9yv1_ELL1H_STIG.gls.par"
        cls.modelJ0613_STIG = model.get_model(cls.parfileJ0613_STIG)

    def test_J1853(self):
        pint_resids_us = Residuals(
            self.toasJ1853, self.modelJ1853, False
        ).time_resids.to(u.s)
        # Due to PINT has higher order of ELL1 model, Tempo2 gives a difference around 3e-8
        # Changed to 4e-8 since modification to get_PSR_freq() makes this 3.1e-8
        log = logging.getLogger("TestJ1853.J1853_residuals")
        diffs = np.abs(pint_resids_us.value - self.ltres)
        log.debug("Diffs: %s\nMax: %s" % (diffs, np.max(diffs)))
        assert np.all(diffs < 4e-8), "J1853 residuals test failed."

    def test_J1853_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelJ1853.binarymodel_delay(self.toasJ1853, None)
        assert np.all(
            np.abs(pint_binary_delay.value + self.ltbindelay) < 3e-8
        ), "J1853 binary delay test failed."

    def test_derivative(self):
        log = logging.getLogger("TestJ1853.derivative_test")
        test_params = ["H3", "H4", "STIGMA"]
        self.modelJ1853.H4.value = 0.0  # For test PBDOT
        self.modelJ1853.STIGMA.value = 0.0
        testp = tdu.get_derivative_params(self.modelJ1853)
        delay = self.modelJ1853.delay(self.toasJ1853)
        # Change parameter test step
        testp["H3"] = 6e-1
        testp["H4"] = 1e-2
        testp["STIGMA"] = 1e-2
        for p in test_params:
            log.debug("Runing derivative for %s", "d_delay_d_" + p)
            ndf = self.modelJ1853.d_phase_d_param_num(self.toasJ1853, p, testp[p])
            adf = self.modelJ1853.d_phase_d_param(self.toasJ1853, delay, p)
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

    def test_J0613_H4(self):
        log = logging.getLogger("TestJ0613.fit_tests")
        f = ff.GLSFitter(self.toasJ0613, self.modelJ0613)
        f.fit_toas()
        f.set_fitparams("H3", "H4")
        for pn, p in (f.get_fitparams()).items():
            op = getattr(f.model_init, pn)
            diff = np.abs(p.value - op.value)
            sigma = diff / op.uncertainty_value
            # Fit th
            assert (
                sigma < 0.7
            ), "refit %s is %lf sigma different from original value" % (pn, sigma)

    def test_J0613_STIG(self):
        log = logging.getLogger("TestJ0613.fit_tests_stig")
        f = ff.GLSFitter(self.toasJ0613, self.modelJ0613_STIG)
        f.fit_toas()
        f.set_fitparams("H3", "STIGMA")
        for pn, p in (f.get_fitparams()).items():
            op = getattr(f.model_init, pn)
            diff = np.abs(p.value - op.value)
            sigma = diff / op.uncertainty_value
            # Fit th
            assert (
                sigma < 0.7
            ), "refit %s is %lf sigma different from original value" % (pn, sigma)


if __name__ == "__main__":
    pass
