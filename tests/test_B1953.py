"""Various tests to assess the performance of the B1953+29."""
import logging
import os
import unittest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
import test_derivative_utils as tdu
from pint.residuals import Residuals
from pinttestdata import datadir


class TestB1953(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""

    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parfileB1953 = "B1953+29_NANOGrav_dfg+12_TAI_FB90.par"
        cls.timB1953 = "B1953+29_NANOGrav_dfg+12.tim"
        cls.toasB1953 = toa.get_TOAs(
            cls.timB1953, ephem="DE405", planets=False, include_bipm=False
        )
        cls.modelB1953 = mb.get_model(cls.parfileB1953)
        # tempo result
        cls.ltres, cls.ltbindelay = np.genfromtxt(
            cls.parfileB1953 + ".tempo2_test", skip_header=1, unpack=True
        )
        print(cls.ltres)

    def test_B1953_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelB1953.binarymodel_delay(self.toasB1953, None)
        assert np.all(
            np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-8
        ), "B1953 binary delay test failed."

    def test_B1953(self):
        pint_resids_us = Residuals(
            self.toasB1953, self.modelB1953, False
        ).time_resids.to(u.s)
        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(
            np.abs(pint_resids_us.value - self.ltres) < 3e-8
        ), "B1953 residuals test failed."

    def test_derivative(self):
        log = logging.getLogger("TestB1953.derivative_test")
        testp = tdu.get_derivative_params(self.modelB1953)
        delay = self.modelB1953.delay(self.toasB1953)
        for p in testp.keys():
            log.debug("Runing derivative for %s", "d_delay_d_" + p)
            ndf = self.modelB1953.d_phase_d_param_num(self.toasB1953, p, testp[p])
            adf = self.modelB1953.d_phase_d_param(self.toasB1953, delay, p)
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


if __name__ == "__main__":
    pass
