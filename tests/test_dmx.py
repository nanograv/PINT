import logging
import os
import unittest

import astropy.units as u
import numpy as np

import pint.toa as toa
from pint import residuals
from pint.models import model_builder as mb
from pinttestdata import datadir


class TestDMX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parf = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_DMX.par")
        cls.timf = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim")
        cls.DMXm = mb.get_model(cls.parf)
        cls.toas = toa.get_TOAs(cls.timf, ephem="DE405", include_bipm=False)

    def test_DMX(self):
        print("Testing DMX module.")
        rs = residuals.Residuals(self.toas, self.DMXm, False).time_resids.to(u.s).value
        ltres, _ = np.genfromtxt(self.parf + ".tempo_test", unpack=True)
        resDiff = rs - ltres
        assert np.all(
            np.abs(resDiff) < 2e-8
        ), "PINT and tempo Residual difference is too big."

    def test_derivative(self):
        log = logging.getLogger("DMX.derivative_test")
        p = "DMX_0002"
        log.debug("Runing derivative for %s", "d_delay_d_" + p)
        ndf = self.DMXm.d_delay_d_param_num(self.toas, p)
        adf = self.DMXm.d_delay_d_param(self.toas, p)
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


if __name__ == "__main__":
    pass
