"""Various tests to assess the performance of the FD model."""
import copy
import os
import unittest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.residuals
import pint.toa as toa
from pinttestdata import datadir


class TestFD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parf = "test_FD.par"
        cls.timf = "test_FD.simulate.pint_corrected"
        cls.FDm = mb.get_model(cls.parf)
        cls.toas = toa.get_TOAs(cls.timf, include_bipm=False)
        # libstempo result
        cls.ltres, cls.ltbindelay = np.genfromtxt(cls.parf + ".tempo_test", unpack=True)

    def test_FD(self):
        print("Testing FD module.")
        rs = (
            pint.residuals.Residuals(self.toas, self.FDm, False)
            .time_resids.to(u.s)
            .value
        )
        resDiff = rs - self.ltres
        # NOTE : This prescision is a lower then 1e-7 seconds level, due to some
        # early parks clock corrections are treated differently.
        # TEMPO2: Clock correction = clock0 + clock1 (in the format of general2)
        # PINT : Clock correction = toas.table['flags']['clkcorr']
        # Those two clock correction difference are causing the trouble.
        assert np.all(resDiff < 5e-6), "PINT and tempo Residual difference is too big. "

    def test_inf_freq(self):
        test_toas = copy.deepcopy(self.toas)
        test_toas.table["freq"][0:5] = np.inf * u.MHz
        fd_delay = self.FDm.components["FD"].FD_delay(test_toas)
        assert np.all(
            np.isfinite
        ), "FD component is not handling infinite frequency right."
        assert np.all(
            fd_delay[0:5].value == 0.0
        ), "FD component did not compute infinite frequency delay right"
        d_d_d_fd = self.FDm.d_delay_FD_d_FDX(test_toas, "FD1")
        assert np.all(np.isfinite), (
            "FD component is not handling infinite frequency right when doning"
            + " derivatives."
        )


if __name__ == "__main__":
    pass
