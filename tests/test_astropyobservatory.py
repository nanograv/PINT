import logging
import os
import unittest
import numpy as np
import pint.observatory


class TestAstropyObservatory(unittest.TestCase):
    """ Test fallback from PINT observatories to astropy observatories"""

    @classmethod
    def setUpClass(cls):
        # name and IRTF of an observatory that PINT should know about
        cls.pint_obsname = "gbt"
        cls.pint_irtf = [882589.65, -4924872.32, 3943729.348]
        # name and IRTF of an observatory that only astropy should know about
        cls.astropy_obsname = "keck"
        cls.astropy_irtf = [-5464487.81759887, -2492806.59108569, 2151240.19451846]
        # name of an observatory that none of them should know about
        cls.none_obsname = "not_an_observatory"

        # IRTF coordinate tolerance in m
        cls.irtf_tolerance = 1

        cls.log = logging.getLogger("TestAstropyObservatory")

    def test_astropy_observatory(self):
        """
        try to instantiate the observatory in PINT from astropy and check their IRTF values
        """
        keck = pint.observatory.get_observatory(self.astropy_obsname)
        keck_irtf = [
            keck.earth_location_itrf().x.value,
            keck.earth_location_itrf().y.value,
            keck.earth_location_itrf().z.value,
        ]
        separation = np.sqrt(
            (keck_irtf[0] - self.astropy_irtf[0]) ** 2
            + (keck_irtf[1] - self.astropy_irtf[1]) ** 2
            + (keck_irtf[2] - self.astropy_irtf[2]) ** 2
        )
        msg = "Checking PINT definition for '%s' failed with IRTF separation %.1e m" % (
            self.astropy_obsname,
            separation,
        )
        assert separation < self.irtf_tolerance

    def test_pint_observatory(self):
        """
        try to instantiate the observatory in PINT and check their IRTF values
        """
        gbt = pint.observatory.get_observatory(self.pint_obsname)
        gbt_irtf = [
            gbt.earth_location_itrf().x.value,
            gbt.earth_location_itrf().y.value,
            gbt.earth_location_itrf().z.value,
        ]
        separation = np.sqrt(
            (gbt_irtf[0] - self.pint_irtf[0]) ** 2
            + (gbt_irtf[1] - self.pint_irtf[1]) ** 2
            + (gbt_irtf[2] - self.pint_irtf[2]) ** 2
        )
        msg = "Checking PINT definition for '%s' failed with IRTF separation %.1e m" % (
            self.pint_obsname,
            separation,
        )
        assert separation < self.irtf_tolerance

    def test_missing_observatory(self):
        """
        try to instantiate a missing observatory
        """
        self.assertRaises(KeyError, pint.observatory.get_observatory, self.none_obsname)
