import os
import pytest

import astropy.units as u
import numpy as np
import pint.observatory
import pint.toa as toa

from pinttestdata import datadir


class TestTroposphereDelay:
    MIN_ALT = 5  # the minimum altitude in degrees for testing the delay model

    FLOAT_THRESHOLD = 1e-12  #

    def setup_method(self):
        # parfile = os.path.join(datadir, "J1744-1134.basic.par")
        # ngc = os.path.join(datadir, "NGC6440E")

        self.toas = toa.get_TOAs(os.path.join(datadir, "NGC6440E.tim"))
        self.model = pint.models.get_model(os.path.join(datadir, "NGC6440E.par"))
        self.modelWithTD = pint.models.get_model(os.path.join(datadir, "NGC6440E.par"))
        self.modelWithTD.CORRECT_TROPOSPHERE.value = True

        self.toasInvalid = toa.get_TOAs(os.path.join(datadir, "NGC6440E.tim"))

        for i in range(len(self.toasInvalid.table)):
            # adjust the timing by half a day to make them invalid
            self.toasInvalid.table["mjd"][i] += 0.5

        self.testAltitudes = np.arange(self.MIN_ALT, 90, 100) * u.deg
        self.testHeights = np.array([10, 100, 1000, 5000]) * u.m

        self.td = self.modelWithTD.components["TroposphereDelay"]

    def test_altitudes(self):
        # the troposphere delay should strictly decrease with increasing altitude above horizon
        delays = self.td.delay_model(self.testAltitudes, 0, 1000 * u.m, 0)

        for i in range(1, len(delays)):
            assert delays[i] < delays[i - 1]

    def test_heights(self):
        # higher elevation observatories should have less troposphere delay
        heightDelays = [
            self.td.delay_model(self.testAltitudes, 0, h, 0) for h in self.testHeights
        ]

        for i in range(1, len(self.testHeights)):
            print(heightDelays[i], heightDelays[i - 1])
            assert np.all(np.less(heightDelays[i], heightDelays[i - 1]))

    def test_model_access(self):
        # make sure that the model components are linked correctly to the troposphere delay
        assert hasattr(self.model, "CORRECT_TROPOSPHERE")
        assert "TroposphereDelay" in self.modelWithTD.components.keys()

        # the model should have the sky coordinates defined
        assert self.td._get_target_skycoord() is not None

    def test_invalid_altitudes(self):
        assert np.all(
            np.less_equal(
                np.abs(self.td.troposphere_delay(self.toasInvalid)),
                self.FLOAT_THRESHOLD * u.s,
            )
        )

    def test_latitude_index(self):
        # the code relies on finding the neighboring latitudes to any site
        # for atmospheric constants defined at every 15 degrees
        # so I will test the nearest neighbors function

        l1 = self.td._find_latitude_index(20 * u.deg)
        l2 = self.td._find_latitude_index(-80 * u.deg)
        l3 = self.td._find_latitude_index(0 * u.deg)
        l4 = self.td._find_latitude_index(-90 * u.deg)

        assert self.td.LAT[l1] <= 20 * u.deg < self.td.LAT[l1 + 1]
        assert self.td.LAT[l2] <= 80 * u.deg <= self.td.LAT[l2 + 1]
        assert self.td.LAT[l3] <= 0 * u.deg <= self.td.LAT[l3 + 1]
        assert self.td.LAT[l4] <= 90 * u.deg <= self.td.LAT[l4 + 1]
