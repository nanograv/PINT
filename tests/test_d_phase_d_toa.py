import os
import unittest

import numpy as np
from astropy._erfa import DJM0

import pint.toa as toa
from pint.models import model_builder as mb
from pint.polycos import Polycos
from pinttestdata import datadir, testdir


class TestD_phase_d_toa(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.chdir(datadir)
        self.parfileB1855 = "B1855+09_polycos.par"
        self.timB1855 = "B1855_polyco.tim"
        self.toasB1855 = toa.get_TOAs(
            self.timB1855, ephem="DE405", planets=False, include_bipm=False
        )
        self.modelB1855 = mb.get_model(self.parfileB1855)
        # Read tempo style polycos.
        self.plc = Polycos()
        self.plc.read_polyco_file("B1855_polyco.dat", "tempo")

    def TestD_phase_d_toa(self):
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
        assert np.all(relative_diff < 1e-8), "d_phae_d_toa test filed."


if __name__ == "__main__":
    pass
