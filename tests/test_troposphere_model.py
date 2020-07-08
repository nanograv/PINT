#!/usr/bin/env python

import os
import unittest

import astropy.units as u
import numpy as np
import pint.models as models
import pint.observatory
import pint.toa as toa
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from pint.models.troposphere_delay import TroposphereDelay
from pint.observatory import get_observatory


class TestTroposphereDelay(unittest.TestCase):

    MIN_ALT = 5  # the minimum altitude in degrees for testing the delay model

    FLOAT_THRESHOLD = 1e-12  #

    def setUp(self):
        ngc = "NGC6440E"

        self.toas = toa.get_TOAs(ngc + ".tim")
        self.model = pint.models.get_model(ngc + ".par")
        self.modelWithTD = pint.models.get_model(ngc + ".par")
        self.modelWithTD.CORRECT_TROPOSPHERE.value = True

        self.toasInvalid = toa.get_TOAs(ngc + ".tim")

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
        heightDelays = []  # list of the delays at each height
        for h in self.testHeights:
            heightDelays.append(self.td.delay_model(self.testAltitudes, 0, h, 0))
        for i in range(1, len(self.testHeights)):
            print(heightDelays[i], heightDelays[i - 1])
            assert np.all(np.less(heightDelays[i], heightDelays[i - 1]))

    def test_model_access(self):
        # make sure that the model components are linked correctly to the troposphere delay
        assert hasattr(self.model, "CORRECT_TROPOSPHERE")

        # the model should have the sky coordinates defined
        assert self.td._get_target_skycoord() is not None

    def test_invalid_altitudes(self):
        assert np.all(
            np.less_equal(
                np.abs(self.td.troposphere_delay(self.toasInvalid)),
                self.FLOAT_THRESHOLD * u.s,
            )
        )
