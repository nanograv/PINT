"""Various tests to assess the performance of the B1855+09."""
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


class TestB1855:
    """Compare delays from the dd model with tempo and PINT"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.parfileB1855 = "B1855+09_NANOGrav_9yv1.gls.par"
        cls.timB1855 = "B1855+09_NANOGrav_9yv1.tim"
        cls.toasB1855 = toa.get_TOAs(
            cls.timB1855, ephem="DE421", planets=False, include_bipm=False
        )
        cls.modelB1855 = mb.get_model(cls.parfileB1855)
        # tempo result
        cls.ltres = np.genfromtxt(
            f"{cls.parfileB1855}.tempo2_test", skip_header=1, unpack=True
        )

    def test_b1855(self):
        pint_resids_us = Residuals(
            self.toasB1855, self.modelB1855, use_weighted_mean=False
        ).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(
            np.abs(pint_resids_us.value - self.ltres) < 3e-8
        ), "B1855 residuals test failed."

    def test_derivative(self):
        log = logging.getLogger("TestB1855.derivative_test")
        testp = tdu.get_derivative_params(self.modelB1855)
        delay = self.modelB1855.delay(self.toasB1855)
        for p in testp.keys():
            log.debug("Runing derivative for %s", f"d_delay_d_{p}")
            ndf = self.modelB1855.d_phase_d_param_num(self.toasB1855, p, testp[p])
            adf = self.modelB1855.d_phase_d_param(self.toasB1855, delay, p)
            diff = adf - ndf
            if np.all(diff.value) != 0.0:
                mean_der = (adf + ndf) / 2.0
                relative_diff = np.abs(diff) / np.abs(mean_der)
                # print "Diff Max is :", np.abs(diff).max()
                msg = (
                    "Derivative test failed at d_delay_d_%s with max relative difference %lf"
                    % (p, np.nanmax(relative_diff).value)
                )
                tol = 0.7 if p in ["SINI"] else 1e-3
                log.debug(
                    (
                        "derivative relative diff for %s, %lf"
                        % (f"d_delay_d_{p}", np.nanmax(relative_diff).value)
                    )
                )
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue
