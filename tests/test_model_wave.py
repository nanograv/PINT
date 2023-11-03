import os
import pytest

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


class TestWave:
    @classmethod
    def setup_class(cls):
        cls.m = pint.models.get_model(parfile)
        cls.t = pint.toa.get_TOAs(timfile, ephem="DE405", include_bipm=False)

    def test_vela_rms_is_small_enough(self):
        rs = pint.residuals.Residuals(self.t, self.m).time_resids
        rms = rs.to(u.us).std()
        assert rms < 350.0 * u.us
