"""Tests for jump model component """
import logging
import os
import unittest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir


class TestJUMP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parf = "B1855+09_NANOGrav_dfg+12_TAI.par"
        cls.timf = "B1855+09_NANOGrav_dfg+12.tim"
        cls.JUMPm = mb.get_model(cls.parf)
        cls.toas = toa.get_TOAs(
            cls.timf, ephem="DE405", planets=False, include_bipm=False
        )
        # libstempo calculation
        cls.ltres = np.genfromtxt(
            cls.parf + ".tempo_test", unpack=True, names=True, dtype=np.longdouble
        )

    def test_jump(self):
        presids_s = Residuals(self.toas, self.JUMPm, False).time_resids.to(u.s)
        assert np.all(
            np.abs(presids_s.value - self.ltres["residuals"]) < 1e-7
        ), "JUMP test failed."

    def test_derivative(self):
        log = logging.getLogger("Jump phase test")
        p = "JUMP2"
        log.debug("Runing derivative for %s", "d_delay_d_" + p)
        ndf = self.JUMPm.d_phase_d_param_num(self.toas, p)
        adf = self.JUMPm.d_phase_d_param(self.toas, self.JUMPm.delay(self.toas), p)
        diff = adf - ndf
        if not np.all(diff.value) == 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_phase_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            assert np.nanmax(relative_diff) < 0.001, msg


if __name__ == "__main__":
    pass
