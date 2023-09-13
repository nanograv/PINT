import os
import pytest

import numpy as np

from erfa import DJM0

import pint.toa as toa
from pint.models import model_builder as mb
from pint.polycos import Polycos
from pinttestdata import datadir, testdir


class TestD_phase_d_toa:
    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.parfileB1855 = "B1855+09_polycos.par"
        cls.timB1855 = "B1855_polyco.tim"
        cls.toasB1855 = toa.get_TOAs(
            cls.timB1855, ephem="DE405", planets=False, include_bipm=False
        )
        cls.modelB1855 = mb.get_model(cls.parfileB1855)
        # Read tempo style polycos.
        cls.plc = Polycos().read("B1855_polyco.dat", "tempo")

    def test_d_phase_d_toa(self):
        pint_d_phase_d_toa = self.modelB1855.d_phase_d_toa(self.toasB1855)
        mjd = np.array(
            [
                np.longdouble(t.jd1 - DJM0) + np.longdouble(t.jd2)
                for t in self.toasB1855.table["mjd"]
            ]
        )
        tempo_d_phase_d_toa = self.plc.eval_spin_freq(mjd)
        diff = pint_d_phase_d_toa.value - tempo_d_phase_d_toa
        relative_diff = diff / tempo_d_phase_d_toa
        assert np.all(np.abs(relative_diff) < 1e-7), "d_phase_d_toa test failed."
