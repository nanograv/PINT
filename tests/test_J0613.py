"""Various tests to assess the performance of the J0623-0200."""
import logging
import os
import pytest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
import test_derivative_utils as tdu
from pint.residuals import Residuals
from pinttestdata import datadir


class TestJ0613:
    """Compare delays from the ELL1 model with tempo and PINT"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.parfileJ0613 = "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par"
        cls.timJ0613 = "J0613-0200_NANOGrav_dfg+12.tim"
        cls.toasJ0613 = toa.get_TOAs(
            cls.timJ0613, ephem="DE405", planets=False, include_bipm=False
        )
        cls.modelJ0613 = mb.get_model(cls.parfileJ0613)
        # tempo result
        cls.ltres, cls.ltbindelay = np.genfromtxt(
            f"{cls.parfileJ0613}.tempo2_test", skip_header=1, unpack=True
        )
        print(cls.ltres)

    def test_j0613_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelJ0613.binarymodel_delay(self.toasJ0613, None)
        assert np.all(
            np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-8
        ), "J0613 binary delay test failed."

    def test_j0613(self):
        pint_resids_us = Residuals(
            self.toasJ0613, self.modelJ0613, use_weighted_mean=False
        ).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(
            np.abs(pint_resids_us.value - self.ltres) < 3e-8
        ), "J0613 residuals test failed."

    def test_derivative(self):
        log = logging.getLogger("TestJ0613.derivative_test")
        self.modelJ0613.PBDOT.value = 0.0  # For test PBDOT
        self.modelJ0613.EPS1DOT.value = 0.0
        self.modelJ0613.EPS2DOT.value = 0.0
        self.modelJ0613.A1DOT.value = 0.0
        testp = tdu.get_derivative_params(self.modelJ0613)
        delay = self.modelJ0613.delay(self.toasJ0613)
        # Change parameter test step
        testp["EPS1"] = 1
        testp["EPS2"] = 1
        testp["PMDEC"] = 1
        testp["PMRA"] = 1
        for p in testp.keys():
            log.debug("Runing derivative for %s", f"d_delay_d_{p}")
            ndf = self.modelJ0613.d_phase_d_param_num(self.toasJ0613, p, testp[p])
            adf = self.modelJ0613.d_phase_d_param(self.toasJ0613, delay, p)
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
