"""Various tests to assess the performance of the DD model."""
import os
import unittest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir


class TestDD(unittest.TestCase):
    """Compare delays from the dd model with libstempo and PINT"""

    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parfileB1855 = "B1855+09_NANOGrav_dfg+12_modified_DD.par"
        cls.timB1855 = "B1855+09_NANOGrav_dfg+12.tim"
        cls.toasB1855 = toa.get_TOAs(
            cls.timB1855, ephem="DE405", planets=False, include_bipm=False
        )
        cls.modelB1855 = mb.get_model(cls.parfileB1855)
        # libstempo result
        cls.ltres, cls.ltbindelay = np.genfromtxt(
            cls.parfileB1855 + ".tempo_test", unpack=True
        )

    def test_J1855_binary_delay(self):
        # Calculate delays with PINT
        pint_binary_delay = self.modelB1855.binarymodel_delay(self.toasB1855, None)
        assert np.all(
            np.abs(pint_binary_delay.value + self.ltbindelay) < 1e-11
        ), "DD B1855 TEST FAILED"

    # TODO: PINT can still incresase the precision by adding more components
    def test_B1855(self):
        pint_resids_us = Residuals(
            self.toasB1855, self.modelB1855, use_weighted_mean=False
        ).time_resids.to(u.s)
        assert np.all(
            np.abs(pint_resids_us.value - self.ltres) < 1e-7
        ), "DD B1855 TEST FAILED"

    def test_stand_alone_model_params(self):
        m = self.modelB1855
        for binary_par in m.binary_instance.binary_params:
            standalone_par = getattr(m.binary_instance, binary_par)
            try:
                pint_par_name = m.match_param_aliases(binary_par)
            except ValueError:
                if binary_par in m.internal_params:
                    pint_par_name = binary_par
                else:
                    pint_par_name = None
            if pint_par_name is None:
                continue
            pint_par = getattr(m, pint_par_name)
            if pint_par.value is not None:
                if hasattr(standalone_par, 'value'):
                    assert pint_par.value == standalone_par.value
                else:
                    assert pint_par.value == standalone_par


if __name__ == "__main__":
    pass
