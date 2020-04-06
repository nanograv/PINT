"""Various tests to assess the performance of the DD model."""
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


class TestDDK(unittest.TestCase):
    """Compare delays from the dd model with libstempo and PINT"""

    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parfileJ1713 = "J1713+0747_NANOGrav_11yv0.gls.par"
        cls.timJ1713 = "J1713+0747_NANOGrav_11yv0_short.tim"
        cls.toasJ1713 = toa.get_TOAs(cls.timJ1713, ephem="DE421", planets=False)
        cls.toasJ1713.table.sort("index")
        cls.modelJ1713 = mb.get_model(cls.parfileJ1713)
        # libstempo result
        cls.ltres, cls.ltbindelay = np.genfromtxt(
            cls.parfileJ1713 + ".tempo_test", unpack=True
        )

    def test_J1713_binary_delay(self):
        # Calculate delays with PINT
        # NOTE tempo and PINT has different definition of parameter KOM. So lower the
        # threshold
        pint_binary_delay = self.modelJ1713.binarymodel_delay(self.toasJ1713, None)
        assert np.all(
            np.abs(pint_binary_delay.value + self.ltbindelay) < 5e-7
        ), "DDK J1713 TEST FAILED"

    def test_J1713(self):
        log = logging.getLogger("TestJ1713.test_J1713")
        pint_resids_us = Residuals(
            self.toasJ1713, self.modelJ1713, False
        ).time_resids.to(u.s)
        diff = pint_resids_us.value - self.ltres
        log.debug("Max diff %lf" % np.abs(diff - diff.mean()).max())
        assert np.all(np.abs(diff - diff.mean()) < 5e-7), "DDK J1713 TEST FAILED"

    def test_J1713_deriv(self):
        log = logging.getLogger("TestJ1713.derivative_test")
        testp = tdu.get_derivative_params(self.modelJ1713)
        delay = self.modelJ1713.delay(self.toasJ1713)
        for p in testp.keys():
            # Only check the binary parameters
            if p not in self.modelJ1713.binary_instance.binary_params:
                continue
            if p in ["PX", "PMRA", "PMDEC"]:
                continue
            par = getattr(self.modelJ1713, p)
            if type(par).__name__ is "boolParameter":
                continue
            log.debug("Runing derivative for %s", "d_phase_d_" + p)
            ndf = self.modelJ1713.d_phase_d_param_num(self.toasJ1713, p, testp[p])
            adf = self.modelJ1713.d_phase_d_param(self.toasJ1713, delay, p)
            diff = adf - ndf
            if not np.all(diff.value) == 0.0:
                mean_der = (adf + ndf) / 2.0
                relative_diff = np.abs(diff) / np.abs(mean_der)
                # print "Diff Max is :", np.abs(diff).max()
                msg = (
                    "Derivative test failed at d_phase_d_%s with max relative difference %lf"
                    % (p, np.nanmax(relative_diff).value)
                )
                if p in ["SINI", "KIN"]:
                    tol = 0.7
                elif p in ["KOM"]:
                    tol = 0.04
                else:
                    tol = 1e-3
                log.debug(
                    "derivative relative diff for %s, %lf"
                    % ("d_phase_d_" + p, np.nanmax(relative_diff).value)
                )
                assert np.nanmax(relative_diff) < tol, msg
            else:
                continue

    def test_K96(self):
        log = logging.getLogger("TestJ1713 Switch of K96")
        self.modelJ1713.K96.value = False
        res = Residuals(self.toasJ1713, self.modelJ1713, False).time_resids.to(u.s)
        delay = self.modelJ1713.delay(self.toasJ1713)
        testp = tdu.get_derivative_params(self.modelJ1713)
        for p in testp.keys():
            adf = self.modelJ1713.d_phase_d_param(self.toasJ1713, delay, p)


if __name__ == "__main__":
    pass
