"""Various tests to assess the performance of the DD model."""
import os
import pytest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir
from copy import deepcopy


class TestDD:
    """Compare delays from the dd model with libstempo and PINT"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.parfileB1855 = "B1855+09_NANOGrav_dfg+12_modified_DD.par"
        cls.timB1855 = "B1855+09_NANOGrav_dfg+12.tim"
        cls.toasB1855 = toa.get_TOAs(
            cls.timB1855, ephem="DE405", planets=False, include_bipm=False
        )
        cls.modelB1855 = mb.get_model(cls.parfileB1855)
        # libstempo result
        cls.ltres, cls.ltbindelay = np.genfromtxt(
            f"{cls.parfileB1855}.tempo_test", unpack=True
        )

    def test_j1855_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelB1855.binarymodel_delay(self.toasB1855, None)
        assert np.all(
            np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-11
        ), "DD B1855 TEST FAILED"

    # TODO: PINT can still increase the precision by adding more components
    def test_b1855(self):
        pint_resids_us = Residuals(
            self.toasB1855, self.modelB1855, use_weighted_mean=False
        ).time_resids.to(u.s)
        assert np.all(
            np.abs(pint_resids_us.value - self.ltres) < 1e-7
        ), "DD B1855 TEST FAILED"

    def test_dd_zero_ecc(self):
        m = deepcopy(self.modelB1855)
        m.ECC.value = 0
        m.validate()  # validate will set the following values to 0
        assert m.OM.value == 0 and m.OMDOT.value == 0 and m.EDOT.value == 0
