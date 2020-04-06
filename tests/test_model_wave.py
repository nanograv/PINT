#! /usr/bin/env python
import os
import unittest

import astropy.units as u

import pint.fitter
import pint.models
import pint.residuals
import pint.toa
from pinttestdata import datadir

# Not included in the test here, but as a sanity check I used this same
# ephemeris to phase up Fermi data, and it looks good.

parfile = os.path.join(datadir, "vela_wave.par")
timfile = os.path.join(datadir, "vela_wave.tim")


class TestWave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = pint.models.get_model(parfile)
        cls.t = pint.toa.get_TOAs(timfile, ephem="DE405", include_bipm=False)

    def test_vela(self):
        print("Test RMS of a VELA ephemeris with WAVE parameters.")
        rs = pint.residuals.Residuals(self.t, self.m).time_resids
        rms = rs.to(u.us).std()
        emsg = "RMS of " + str(rms.value) + " is too big."
        assert rms < 350.0 * u.us, emsg


if __name__ == "__main__":
    unittest.main()
