"""Various tests to assess the performance of early CHIME data."""
import pint.models.model_builder as mb
import pint.toa as toa
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest
import test_derivative_utils as tdu
import logging
from pinttestdata import testdir, datadir

os.chdir(datadir)


class Test_CHIME_data(unittest.TestCase):
    """Compare delays from the dd model with tempo and PINT"""
    @classmethod
    def setUpClass(self):
        self.parfile = 'B1937+21.basic.par'
        self.tim = 'B1937+21.CHIME.CHIME.NG.N.tim'

    def test_toa_read(self):
        toas = toa.get_TOAs(self.tim, ephem="DE436", planets=False,
                            include_bipm=True)
        assert toas.ntoas == 848, "CHIME TOAs did not read correctly."
        assert list(set(toas.get_obss())) == ['chime'], \
            "CHIME did not recognized by observatory module."

    def test_residuals(self):
        model = mb.get_model(self.parfile)
        toas = toa.get_TOAs(self.tim, ephem="DE436", planets=False,
                            include_bipm=True)
        r = resids(toas, model)
        assert np.all(np.abs(r.time_resids.to(u.us)) < 800 * u.us), \
            "Residuals did not computed correctly for early CHIME data."
