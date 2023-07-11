import pytest
import logging
import pytest
import numpy as np
import pint.observatory


class TestAstropyObservatory:
    """
    Test fallback from PINT observatories to astropy observatories."""

    @classmethod
    def setup_class(cls):
        # name and ITRF of an observatory that PINT should know about
        cls.pint_obsname = "gbt"
        cls.pint_itrf = [882589.65, -4924872.32, 3943729.348]
        # name and ITRF of an observatory that only astropy should know about
        cls.astropy_obsname = "keck"
        cls.astropy_itrf = [-5464487.81759887, -2492806.59108569, 2151240.19451846]
        # name of an observatory that none of them should know about
        cls.none_obsname = "not_an_observatory"

        # ITRF coordinate tolerance in m
        cls.itrf_tolerance = 1

        cls.log = logging.getLogger("TestAstropyObservatory")

    def test_astropy_observatory(self):
        """
        try to instantiate the observatory in PINT from astropy and check their ITRF values.
        """
        keck = pint.observatory.Observatory.get(self.astropy_obsname)
        keck_itrf = [
            keck.earth_location_itrf().x.value,
            keck.earth_location_itrf().y.value,
            keck.earth_location_itrf().z.value,
        ]
        separation = np.sqrt(
            (keck_itrf[0] - self.astropy_itrf[0]) ** 2
            + (keck_itrf[1] - self.astropy_itrf[1]) ** 2
            + (keck_itrf[2] - self.astropy_itrf[2]) ** 2
        )
        msg = "Checking PINT definition for '%s' failed with ITRF separation %.1e m" % (
            self.astropy_obsname,
            separation,
        )
        assert separation < self.itrf_tolerance, msg

    def test_pint_observatory(self):
        """
        try to instantiate the observatory in PINT and check their ITRF values.
        """
        gbt = pint.observatory.Observatory.get(self.pint_obsname)
        gbt_itrf = [
            gbt.earth_location_itrf().x.value,
            gbt.earth_location_itrf().y.value,
            gbt.earth_location_itrf().z.value,
        ]
        separation = np.sqrt(
            (gbt_itrf[0] - self.pint_itrf[0]) ** 2
            + (gbt_itrf[1] - self.pint_itrf[1]) ** 2
            + (gbt_itrf[2] - self.pint_itrf[2]) ** 2
        )
        msg = "Checking PINT definition for '%s' failed with ITRF separation %.1e m" % (
            self.pint_obsname,
            separation,
        )
        assert separation < self.itrf_tolerance, msg

    def test_missing_observatory(self):
        """
        try to instantiate a missing observatory.
        """
        pytest.raises(KeyError, pint.observatory.Observatory.get, self.none_obsname)
