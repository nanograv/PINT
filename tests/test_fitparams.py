import logging
import os
import unittest

import astropy.units as u

import pint.models as models
from pint import fitter, toa
from pinttestdata import datadir


class TestFitparams(unittest.TestCase):
    """Test whether unfreezing parameters that are not in the model will cause fitting to fail"""

    @classmethod
    def setUpClass(cls):
        # J0613 is in equatorial
        cls.parfile = os.path.join(datadir, "NGC6440E.par")
        cls.model = models.get_model(cls.parfile)

        cls.timfile = os.path.join(datadir, "NGC6440E.tim")
        cls.toas = toa.get_TOAs(cls.timfile, usepickle=False)
        cls.fitter = fitter.WLSFitter(cls.toas, cls.model)
        cls.log = logging.getLogger("TestFitparams")

    def test_fit_params(self):
        """
        unfreeze valid parameters and fit
        unfreeze invalid parameters and fail
        """
        # this should succeed since all of these are valid parameters
        self.fitter.set_fitparams("RAJ", "DECJ", "F0", "F1", "DM")
        self.fitter.fit_toas()

        # this should fail
        self.assertRaises(
            AttributeError,
            self.fitter.set_fitparams,
            ("RAJ", "DECJ", "F0", "F1", "DM", "CAPYBARA"),
        )
